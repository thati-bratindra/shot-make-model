# NBA shot make probability model
# Bratindra Reddy Thati
# ---------------------------------------------------------------------------
# Pipeline
#   1. read training / testing data
#   2. feature engineering (geometry, defender pressure, context) on train+test
#      together so encodings and one-hot columns are identical across sets
#   3. stage-1 GBM with no player/team identity -> out-of-fold expected make
#      probability, used ONLY to measure who beats expectations in step 4
#   4. out-of-fold, shrunk *residual* encodings for shooter / closest defender /
#      shooter-by-shot-type / offensive team / defensive team
#   5. stage-2 GBM on shot features + identity encodings (not the stage-1
#      prediction itself) -> OOF predictions + one model per fold
#   6. smooth spline logistic model, blended with the GBM at the OOF-optimal
#      weight (guards against the GBM being locally over-confident)
#   7. calibration check, clipping, write predictions.csv
#   8. save model_artifacts.rds, which project_writeup.Rmd renders from
# ---------------------------------------------------------------------------

set.seed(20260918)

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(splines)
  library(xgboost)
})

# ------------------------------- settings ----------------------------------

DATA_DIR  <- "data"   # run simulate_data.R first; the script also looks in "." as a fallback
N_FOLDS   <- 5
SAMPLE_N  <- NULL     # e.g. 60000 to debug quickly; NULL = use all training rows
RUN_BLEND <- TRUE     # fit the spline logistic model and blend
CLIP      <- c(0.002, 0.998)   # never output a probability outside this range

find_file <- function(f) {
  cand <- c(file.path(DATA_DIR, f), f)
  hit  <- cand[file.exists(cand)]
  if (!length(hit)) stop("Could not find ", f, " in '", DATA_DIR, "' or working dir")
  hit[1]
}

logloss <- function(y, p) {
  p <- pmin(pmax(p, 1e-15), 1 - 1e-15)
  -mean(y * log(p) + (1 - y) * log(1 - p))
}
logit    <- function(p) log(p / (1 - p))
invlogit <- function(x) 1 / (1 + exp(-x))

# ------------------------------ 1. read data -------------------------------

# contester3/4 and distcont3/4 are almost all empty, so readr's type guessing
# (first 1000 rows) gets them wrong; pin them explicitly.
col_over <- cols(
  shot_id       = col_character(),
  contester3_id = col_character(),
  contester4_id = col_character(),
  distcont3     = col_double(),
  distcont4     = col_double()
)

message("reading data ...")
train <- read_csv(find_file("training.csv.gz"), col_types = col_over, progress = FALSE)
test  <- read_csv(find_file("testing.csv.gz"),  col_types = col_over, progress = FALSE)

cat("train:", nrow(train), "rows |  test:", nrow(test), "rows\n")
cat("base make rate:", round(mean(as.integer(train$outcome)), 4), "\n")

# check on the one column whose text format we have to parse
cat("\nclosestdefapproach examples:\n")
print(utils::head(train$closestdefapproach, 3))

if (!is.null(SAMPLE_N) && SAMPLE_N < nrow(train)) {
  train <- train[sample(nrow(train), SAMPLE_N), ]
  message("*** SAMPLE MODE: using ", nrow(train), " training rows ***")
}

# --------------------------- 2. feature engineering ------------------------

# The file stores these as "{20.047,18.821,16.569,13.862}" - braces, comma
# separated, no spaces. Brackets and space separation also parse, just in case.
parse_approach <- function(x) {
  s <- str_replace_all(as.character(x), "[\\{\\}\\[\\]\"']", "")
  s <- str_trim(s)
  s <- str_replace_all(s, ",\\s*", ",")     # comma + space -> comma
  s <- str_replace_all(s, "\\s+", ",")      # remaining whitespace -> comma
  m <- str_split_fixed(s, ",", 4)
  out <- suppressWarnings(matrix(as.numeric(m), ncol = 4))
  colnames(out) <- c("def_d100", "def_d075", "def_d050", "def_d025")
  out
}

to01 <- function(x) {
  if (is.logical(x)) return(as.integer(x))
  if (is.numeric(x)) return(as.integer(x != 0))
  as.integer(tolower(str_trim(as.character(x))) %in% c("true", "t", "yes", "y", "1"))
}

# The file uses OCT, NOV, DEC, JAN, FEB, MAR, APR. Map to 1..7 so the feature is
# ordered by point in the season rather than alphabetically.
month_index <- function(x) {
  v  <- tolower(str_trim(as.character(x)))
  nm <- c("oct", "nov", "dec", "jan", "feb", "mar", "apr", "may", "jun",
          "jul", "aug", "sep")
  i <- match(substr(v, 1, 3), nm)
  if (all(is.na(i))) i <- as.integer(factor(x))
  as.numeric(i)
}

make_features <- function(df) {
  ap <- parse_approach(df$closestdefapproach)

  # geometry: rim center is at x = -41.75, y = 0
  dx  <- df$locationx + 41.75
  dy  <- df$locationy
  d   <- sqrt(dx^2 + dy^2)
  # 0 deg = straight on, 90 deg = along the baseline (corner)
  ang <- atan2(abs(dy), pmax(dx, 0.01)) * 180 / pi

  cont <- cbind(df$distcont1, df$distcont2, df$distcont3, df$distcont4)
  min_cont  <- do.call(pmin, c(as.list(as.data.frame(cont)), na.rm = TRUE))
  mean_cont <- rowMeans(cont, na.rm = TRUE); mean_cont[is.nan(mean_cont)] <- NA_real_
  n_cont_4  <- rowSums(cont < 4, na.rm = TRUE)
  n_cont_2  <- rowSums(cont < 2, na.rm = TRUE)
  pressure  <- rowSums(1 / (1 + cont), na.rm = TRUE)   # 0 when nobody contests

  cdd <- df$closestdefdist
  win_min <- do.call(pmin, list(ap[, 1], ap[, 2], ap[, 3], ap[, 4], cdd, na.rm = TRUE))

  num <- tibble(
    # shot location
    distance         = df$distance,
    dist_calc        = d,
    dist_resid       = df$distance - d,     # flags rows where the two disagree
    log_dist         = log1p(d),
    angle            = ang,
    abs_y            = abs(dy),
    locationx        = df$locationx,
    locationy        = df$locationy,
    three            = to01(df$three),
    corner_three     = as.integer(to01(df$three) == 1 & abs(dy) > 20),

    # defender nearest the shooter
    closestdefdist   = cdd,
    def_d100         = ap[, 1],
    def_d075         = ap[, 2],
    def_d050         = ap[, 3],
    def_d025         = ap[, 4],
    def_close_1s     = ap[, 1] - cdd,               # ft closed in the last second
    def_close_late   = (ap[, 4] - cdd) / 0.25,      # ft/s over the last 0.25 s
    def_min_window   = win_min,
    def_dist_ratio   = cdd / (d + 1),

    # contesters
    num_contesters   = df$num_contesters,
    distcont1        = df$distcont1,
    distcont2        = df$distcont2,
    distcont3        = df$distcont3,
    distcont4        = df$distcont4,
    min_cont         = min_cont,
    mean_cont        = mean_cont,
    n_cont_4         = n_cont_4,
    n_cont_2         = n_cont_2,
    pressure         = pressure,
    contested        = to01(df$contested),

    # shooter state and clock
    shooterspeed     = df$shooterspeed,
    dribbles         = pmin(df$dribblesbefore, 12),
    no_dribble       = as.integer(df$dribblesbefore == 0),
    many_dribbles    = as.integer(df$dribblesbefore >= 6),
    shotclock        = df$shotclock,
    shotclock_lt4    = as.integer(df$shotclock < 4),
    shotclock_gt18   = as.integer(df$shotclock > 18),

    # season context
    month_idx        = month_index(df$month),
    season_idx       = as.numeric(as.integer(factor(df$season_id)))
  )

  cat_df <- tibble(
    shottype  = as.character(df$shottype),
    gamestate = as.character(df$gamestate)
  )

  list(num = num, cat = cat_df)
}

onehot <- function(cat_df) {
  do.call(cbind, lapply(names(cat_df), function(nm) {
    v <- cat_df[[nm]]
    v[is.na(v)] <- "missing"
    f <- factor(v)
    m <- matrix(0, nrow(cat_df), nlevels(f))
    m[cbind(seq_len(nrow(cat_df)), as.integer(f))] <- 1
    colnames(m) <- paste0(nm, "_", make.names(levels(f)))
    m
  }))
}

message("engineering features ...")
n_tr <- nrow(train)
all_raw <- bind_rows(
  train %>% mutate(.set = "train"),
  test  %>% mutate(.set = "test", outcome = NA_integer_)
)
fe <- make_features(all_raw)

na_rate <- mean(is.na(fe$num$def_d100))
cat("NA rate in parsed closestdefapproach:", round(na_rate, 3), "\n")
if (na_rate > 0.5) warning("closestdefapproach parsed mostly NA - check the raw format above")

X_num  <- as.matrix(fe$num)
X_cat  <- onehot(fe$cat)
X_base <- cbind(X_num, X_cat)          # stage-1 feature block (no identities)
rm(X_num, X_cat); invisible(gc())

y      <- as.integer(train$outcome)   # stored as TRUE/FALSE in the file
tr_idx <- seq_len(n_tr)
te_idx <- (n_tr + 1):nrow(all_raw)
folds  <- sample(rep_len(seq_len(N_FOLDS), n_tr))

# ------------------------------ xgboost helpers ----------------------------

params <- list(
  objective        = "binary:logistic",
  eval_metric      = "logloss",
  eta              = 0.05,
  max_depth        = 6,
  min_child_weight = 50,
  subsample        = 0.8,
  colsample_bytree = 0.8,
  lambda           = 1,
  tree_method      = "hist",
  nthread          = max(1, parallel::detectCores() - 1)
)

fit_one <- function(X, y, tr, va, params, nrounds, early = 50) {
  dtr <- xgb.DMatrix(X[tr, , drop = FALSE], label = y[tr], missing = NA)
  dva <- xgb.DMatrix(X[va, , drop = FALSE], label = y[va], missing = NA)
  args <- list(params = params, data = dtr, nrounds = nrounds,
               early_stopping_rounds = early, verbose = 1, print_every_n = 200)
  # xgboost renamed `watchlist` to `evals` in 2.0
  if (utils::packageVersion("xgboost") >= "2.0.0") args$evals <- list(valid = dva)
  else                                             args$watchlist <- list(valid = dva)
  do.call(xgb.train, args)
}

# xgboost moved best_iteration onto the object's R attributes in 2.1, and from
# that version predict() already uses the best iteration when the model was
# trained with early stopping. Handle both old and new.
XGB_NEW <- utils::packageVersion("xgboost") >= "2.1.0"

best_iter <- function(m) {
  cand <- list(attributes(m)$best_iteration, m$best_iteration)
  for (v in cand) if (!is.null(v) && length(v) == 1 && !is.na(v)) return(as.numeric(v))
  NA_real_
}

pred_best <- function(m, X) {
  d <- xgb.DMatrix(X, missing = NA)
  if (XGB_NEW) return(as.numeric(predict(m, d)))   # honors best_iteration itself
  bi <- best_iter(m)
  if (!is.na(bi)) {
    out <- try(predict(m, d, iterationrange = c(1, bi + 1)), silent = TRUE)
    if (!inherits(out, "try-error")) return(as.numeric(out))
  }
  as.numeric(predict(m, d))
}

run_cv <- function(X, y, folds, params, nrounds) {
  oof    <- rep(NA_real_, length(y))
  models <- vector("list", N_FOLDS)
  for (f in seq_len(N_FOLDS)) {
    message("  fold ", f, "/", N_FOLDS)
    tr <- which(folds != f); va <- which(folds == f)
    m  <- fit_one(X, y, tr, va, params, nrounds)
    oof[va]     <- pred_best(m, X[va, , drop = FALSE])
    models[[f]] <- m
  }
  list(oof = oof, models = models,
       best = vapply(models, best_iter, numeric(1)))
}

# ----------------- 3. stage-1 model: no player identity --------------------

message("stage 1: GBM without identity features ...")
p1 <- params; p1$eta <- 0.08
stage1 <- run_cv(X_base[tr_idx, , drop = FALSE], y, folds, p1, nrounds = 1500)
ll_stage1 <- logloss(y, stage1$oof)
cat("stage-1 OOF log-loss:", round(ll_stage1, 5), "\n")

# Stage 1 is scaffolding: its out-of-fold predictions exist only so step 4 can
# measure how far each player/team beats the shot-level expectation. Nothing
# from stage 1 is scored on the test set, so no test predictions are needed.

# ------------- 4. out-of-fold shrunk residual encodings --------------------
# For each identity we store sum(y - p_stage1) and n, then read off
#     effect = sum_residual / (n + k)
# leaving out the fold being predicted. This is an empirical-Bayes shrunk
# estimate of "how much better/worse than the shot-level baseline is this
# player", so it is not just a raw make rate confounded with shot selection.

resid_encode <- function(id_tr, resid, folds, id_te, k) {
  d <- tibble(id = as.character(id_tr), r = resid, f = folds)
  agg <- d %>% group_by(id, f) %>% summarise(s = sum(r), n = n(), .groups = "drop")
  tot <- agg %>% group_by(id) %>% summarise(S = sum(s), N = sum(n), .groups = "drop")
  agg <- agg %>% left_join(tot, by = "id") %>%
    mutate(e = (S - s) / (N - n + k))     # leave-this-fold-out effect

  key_tr <- paste(d$id, d$f, sep = "|")
  key_ag <- paste(agg$id, agg$f, sep = "|")
  e_tr   <- agg$e[match(key_tr, key_ag)]

  tot <- tot %>% mutate(e_full = S / (N + k), n_log = log1p(N))
  j   <- match(as.character(id_te), tot$id)
  list(train = e_tr,
       test  = ifelse(is.na(j), 0, tot$e_full[j]),
       n_train = log1p(ifelse(is.na(match(d$id, tot$id)), 0, tot$N[match(d$id, tot$id)])),
       n_test  = ifelse(is.na(j), 0, tot$n_log[j]))
}

message("building identity encodings ...")
resid1 <- y - stage1$oof

enc_specs <- list(
  shooter     = list(tr = train$shooter_id,   te = test$shooter_id,   k = 150),
  closestdef  = list(tr = train$closestdef_id, te = test$closestdef_id, k = 300),
  off_team    = list(tr = train$off_team_id,  te = test$off_team_id,  k = 800),
  def_team    = list(tr = train$def_team_id,  te = test$def_team_id,  k = 800),
  shooter_typ = list(tr = paste(train$shooter_id, train$shottype, to01(train$three)),
                     te = paste(test$shooter_id,  test$shottype,  to01(test$three)),
                     k = 250)
)

enc_tr <- list(); enc_te <- list()
for (nm in names(enc_specs)) {
  s <- enc_specs[[nm]]
  e <- resid_encode(s$tr, resid1, folds, s$te, s$k)
  enc_tr[[paste0("enc_", nm)]]  <- e$train
  enc_te[[paste0("enc_", nm)]]  <- e$test
  enc_tr[[paste0("n_", nm)]]    <- e$n_train
  enc_te[[paste0("n_", nm)]]    <- e$n_test
}
enc_tr <- as.matrix(as.data.frame(enc_tr))
enc_te <- as.matrix(as.data.frame(enc_te))

# Stage 2 gets the raw shot features plus the encodings, but NOT the stage-1
# prediction itself. An earlier version stacked it in as a feature; it took
# ~50% of stage 2's importance (stage 2 mostly copied stage 1), and it was
# mismatched between sets - on training rows it was a single out-of-fold
# prediction, on test rows an average of five models, which is smoother.
# Stage 2 has every raw feature, so it can rebuild shot quality itself.
X_tr <- cbind(X_base[tr_idx, , drop = FALSE], enc_tr)
X_te <- cbind(X_base[te_idx, , drop = FALSE], enc_te)
rm(X_base); invisible(gc())

# ----------------- 5. stage-2 model: everything ----------------------------

message("stage 2: GBM with identity encodings ...")
stage2 <- run_cv(X_tr, y, folds, params, nrounds = 3000)
oof_gbm <- stage2$oof
ll_gbm  <- logloss(y, oof_gbm)
cat("stage-2 OOF log-loss:", round(ll_gbm, 5),
    " (stage 1:", round(ll_stage1, 5), ")\n")

pred_gbm_test <- rowMeans(vapply(stage2$models, function(m) pred_best(m, X_te),
                                 numeric(nrow(X_te))))

fold_ll <- vapply(seq_len(N_FOLDS),
                  function(f) logloss(y[folds == f], oof_gbm[folds == f]), numeric(1))

# ----------------- 6. smooth spline logistic model + blend -----------------
# The GBM wins on accuracy, but a smooth parametric model is steadier in
# thin regions of the feature space (heaves, very long closeouts).

oof_glm <- rep(NA_real_, n_tr); pred_glm_test <- rep(NA_real_, nrow(X_te))
w_best  <- 1
ll_glm  <- NA_real_; ll_blend <- NA_real_

if (RUN_BLEND) {
  message("fitting spline logistic model ...")
  med <- function(v) { v[is.na(v)] <- median(v, na.rm = TRUE); v }
  gd <- tibble(
    outcome   = c(y, rep(NA_integer_, nrow(X_te))),
    dist      = med(fe$num$dist_calc),
    angle     = med(fe$num$angle),
    cdd       = med(fe$num$closestdefdist),
    clock     = med(fe$num$shotclock),
    speed     = med(fe$num$shooterspeed),
    dribbles  = med(fe$num$dribbles),
    no_dribble = factor(fe$num$no_dribble),
    pressure  = med(fe$num$pressure),
    enc_sh    = c(enc_tr[, "enc_shooter"], enc_te[, "enc_shooter"]),
    shottype  = factor(ifelse(is.na(fe$cat$shottype), "missing", fe$cat$shottype)),
    gamestate = factor(ifelse(is.na(fe$cat$gamestate), "missing", fe$cat$gamestate)),
    three     = factor(fe$num$three)
  )
  # Terms kept additive on purpose. shottype * three is perfectly separated
  # (zero three-point dunks or tips, 4 three-point layups) and ns(dist) * three
  # is near-aliased, since 2PT attempts stop at ~23.7 ft and 3PT start at 22 ft;
  # both made glm.fit diverge. Dribbles use log1p plus a catch-and-shoot flag
  # rather than a spline, because 46% of shots have zero dribbles and the
  # quantile knots collapsed onto the boundary. Interactions are the GBM's job.
  form <- outcome ~ ns(dist, 6) + ns(angle, 4) + ns(cdd, 5) +
    ns(clock, 4) + ns(speed, 4) + log1p(dribbles) + no_dribble +
    ns(pressure, 4) + shottype + three + gamestate + enc_sh

  for (f in seq_len(N_FOLDS)) {
    message("  fold ", f, "/", N_FOLDS)
    tr <- which(folds != f); va <- which(folds == f)
    g  <- glm(form, data = gd[tr, ], family = binomial())
    if (!g$converged) warning("glm did not converge on fold ", f)
    if (any(is.na(coef(g)))) warning("aliased coefficients on fold ", f, ": ",
                                     paste(names(coef(g))[is.na(coef(g))], collapse = ", "))
    oof_glm[va] <- predict(g, newdata = gd[va, ], type = "response")
  }
  g_full <- glm(form, data = gd[tr_idx, ], family = binomial())
  pred_glm_test <- predict(g_full, newdata = gd[te_idx, ], type = "response")

  ll_glm <- logloss(y, oof_glm)
  ws <- seq(0, 1, by = 0.05)
  lls <- vapply(ws, function(w)
    logloss(y, invlogit(w * logit(pmin(pmax(oof_gbm, 1e-6), 1 - 1e-6)) +
                          (1 - w) * logit(pmin(pmax(oof_glm, 1e-6), 1 - 1e-6)))),
    numeric(1))
  w_best   <- ws[which.min(lls)]
  ll_blend <- min(lls)
  cat("spline OOF log-loss:", round(ll_glm, 5),
      "| blend weight on GBM:", w_best,
      "| blended OOF log-loss:", round(ll_blend, 5), "\n")
}

blend <- function(a, b, w) {
  invlogit(w * logit(pmin(pmax(a, 1e-6), 1 - 1e-6)) +
             (1 - w) * logit(pmin(pmax(b, 1e-6), 1 - 1e-6)))
}
oof_final  <- if (RUN_BLEND) blend(oof_gbm, oof_glm, w_best) else oof_gbm
pred_final <- if (RUN_BLEND) blend(pred_gbm_test, pred_glm_test, w_best) else pred_gbm_test

# ----------------- 7. calibration, clipping, predictions -------------------

cal <- tibble(p = oof_final, y = y) %>%
  mutate(bin = ntile(p, 20)) %>%
  group_by(bin) %>%
  summarise(pred = mean(p), actual = mean(y), n = n(), .groups = "drop")
print(as.data.frame(cal), digits = 4)

# Platt check: if the OOF logit slope is materially off 1, rescale.
platt <- glm(y ~ logit(pmin(pmax(oof_final, 1e-6), 1 - 1e-6)), family = binomial())
platt_pred <- as.numeric(predict(platt, type = "response"))
ll_platt <- logloss(y, platt_pred)
cat("Platt coefficients:", round(coef(platt), 4),
    "| OOF log-loss after rescale:", round(ll_platt, 5), "\n")

use_platt <- (logloss(y, oof_final) - ll_platt) > 2e-4
if (use_platt) {
  message("applying Platt rescaling")
  cf <- coef(platt)
  rescale <- function(p) invlogit(cf[1] + cf[2] * logit(pmin(pmax(p, 1e-6), 1 - 1e-6)))
  oof_final  <- rescale(oof_final)
  pred_final <- rescale(pred_final)
}

pred_final <- pmin(pmax(pred_final, CLIP[1]), CLIP[2])
ll_final   <- logloss(y, pmin(pmax(oof_final, CLIP[1]), CLIP[2]))
cat("final OOF log-loss:", round(ll_final, 5),
    "| intercept-only baseline:", round(logloss(y, rep(mean(y), n_tr)), 5), "\n")

sub <- tibble(shot_id = test$shot_id, make_prob = pred_final)
stopifnot(!any(is.na(sub$make_prob)),
          all(sub$make_prob > 0), all(sub$make_prob < 1))
write_csv(sub, "predictions.csv")
cat("wrote predictions.csv:", nrow(sub), "rows | mean prediction:",
    round(mean(sub$make_prob), 4), "\n")

# ----------------- 8. artifacts for the writeup ----------------------------

imp <- lapply(stage2$models, function(m) {
  out <- try(xgb.importance(model = m, feature_names = colnames(X_tr)), silent = TRUE)
  if (inherits(out, "try-error")) out <- xgb.importance(model = m)  # 2.1+ signature
  out
})
imp <- bind_rows(imp) %>%
  group_by(Feature) %>%
  summarise(Gain = mean(Gain), Cover = mean(Cover), .groups = "drop") %>%
  arrange(desc(Gain))
print(as.data.frame(utils::head(imp, 25)), digits = 4)

# SHAP values on a sample, kept small so the rds stays well under 10 MB
set.seed(1)
shap_rows <- sample(n_tr, min(6000, n_tr))
dshap <- xgb.DMatrix(X_tr[shap_rows, , drop = FALSE], missing = NA)
sv <- try(predict(stage2$models[[1]], dshap, predcontrib = TRUE), silent = TRUE)
if (inherits(sv, "try-error"))
  sv <- predict(stage2$models[[1]], dshap, type = "contrib")   # 2.1+ argument
colnames(sv) <- c(colnames(X_tr), "BIAS")
top_feats <- utils::head(imp$Feature, 10)
shap_long <- bind_rows(lapply(top_feats, function(f) tibble(
  feature = f,
  shap    = as.numeric(sv[, f]),
  value   = as.numeric(X_tr[shap_rows, f])
)))

effect_curves <- bind_rows(
  tibble(var = "shot distance (ft)", x = fe$num$dist_calc[tr_idx],
         p = oof_final, y = y) %>%
    mutate(bin = cut(x, breaks = c(seq(0, 30, 2), 40, 100), include.lowest = TRUE)),
  tibble(var = "closest defender (ft)", x = fe$num$closestdefdist[tr_idx],
         p = oof_final, y = y) %>%
    mutate(bin = cut(x, breaks = c(seq(0, 10, 1), 15, 40), include.lowest = TRUE)),
  tibble(var = "shot clock (s)", x = fe$num$shotclock[tr_idx],
         p = oof_final, y = y) %>%
    mutate(bin = cut(x, breaks = c(0, 2, 4, 6, 8, 10, 14, 18, 24), include.lowest = TRUE))
) %>%
  filter(!is.na(bin)) %>%
  group_by(var, bin) %>%
  summarise(x = mean(x, na.rm = TRUE), pred = mean(p), actual = mean(y),
            n = n(), .groups = "drop")

by_type <- tibble(shottype = fe$cat$shottype[tr_idx], p = oof_final, y = y) %>%
  group_by(shottype) %>%
  summarise(n = n(), pred = mean(p), actual = mean(y), .groups = "drop") %>%
  arrange(desc(n))

artifacts <- list(
  n_train    = n_tr,
  n_test     = nrow(test),
  base_rate  = mean(y),
  importance = imp,
  shap_long  = shap_long,
  calibration = cal,
  effects    = effect_curves,
  by_type    = by_type,
  fold_ll    = fold_ll,
  scores     = tibble(
    model = c("intercept only", "stage 1 GBM (no identity)", "stage 2 GBM",
              "spline logistic", "blend (final)"),
    oof_logloss = c(logloss(y, rep(mean(y), n_tr)), ll_stage1, ll_gbm, ll_glm, ll_final)
  ),
  blend_weight = w_best,
  platt_applied = use_platt,
  best_rounds = stage2$best,
  params = params,
  pred_summary = summary(sub$make_prob),
  n_features = ncol(X_tr)
)
saveRDS(artifacts, "model_artifacts.rds")
message("saved model_artifacts.rds - now knit writeup.Rmd")
