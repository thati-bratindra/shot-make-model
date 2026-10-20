# Synthetic shot data
# ---------------------------------------------------------------------------
# Writes data/training.csv.gz and data/testing.csv.gz with the same columns,
# types and formats the model code reads. Every team, player and shot is
# invented: players get hidden shooting and defending skills, shots are drawn
# from simple location / defender / clock models, and each outcome is a coin
# flip with a probability built from those pieces. Because the true skills are
# known, the data can be used to check how much of them the model recovers.
#
# A few quirks are kept on purpose because the model code relies on or guards
# against them: no three-point dunks or tip-ins and exactly four three-point
# layups, a large share of zero-dribble shots, and contester3/4 columns that
# are empty in almost every row.
# ---------------------------------------------------------------------------

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
})

set.seed(20261019)

N_TRAIN <- 200000   # two seasons
N_TEST  <- 100000   # a third season, predicted by the model
N_TEAMS <- 30
ROSTER  <- 15
RIM_X   <- -41.75

hex_ids <- function(n, width) {
  out <- character(0)
  while (length(out) < n) {
    cand <- vapply(seq_len(n), function(i)
      paste(sample(c(0:9, letters[1:6]), width, replace = TRUE), collapse = ""), "")
    # keep ids that contain a letter other than "e", so no reader parses them as numbers
    out <- unique(c(out, cand[grepl("[a-df]", cand)]))
  }
  out[seq_len(n)]
}

seasons <- hex_ids(3, 5)
teams   <- tibble(team_id = hex_ids(N_TEAMS, 6),
                  off_skill = rnorm(N_TEAMS, 0, 0.04),
                  def_skill = rnorm(N_TEAMS, 0, 0.05))   # positive = opponents shoot better

# ---- players: a pool that turns over a little each season ------------------
n_pool  <- 820
players <- tibble(
  player_id  = hex_ids(n_pool, 10),
  role       = sample(c("guard", "wing", "big"), n_pool, TRUE, c(0.4, 0.35, 0.25)),
  usage      = rlnorm(n_pool, 0, 0.9),
  jump_skill = rnorm(n_pool, 0, 0.22),
  rim_skill  = rnorm(n_pool, 0, 0.20),
  def_skill  = rnorm(n_pool, 0, 0.12)                    # positive = shooters do better
) %>%
  mutate(three_skill = 0.6 * jump_skill + rnorm(n_pool, 0, 0.18),
         jump_skill  = ifelse(role == "big", jump_skill - 0.15, jump_skill))

# season rosters: 450 players per season, ~25% change teams between seasons,
# and new players replace retirements, so some test-season players are unseen
make_rosters <- function() {
  pool_idx <- seq_len(n_pool)
  active <- list(sample(pool_idx, N_TEAMS * ROSTER))
  for (s in 2:3) {
    keep  <- sample(active[[s - 1]], round(0.9 * length(active[[s - 1]])))
    fresh <- sample(setdiff(pool_idx, unlist(active)), N_TEAMS * ROSTER - length(keep))
    active[[s]] <- c(keep, fresh)
  }
  bind_rows(lapply(1:3, function(s) {
    idx  <- active[[s]]
    team <- rep(seq_len(N_TEAMS), each = ROSTER)[order(runif(length(idx)))]
    tibble(season = s, p = idx, team = team)
  }))
}
rosters <- make_rosters()
# keep ~75% of returning players on the same team as last season
for (s in 2:3) {
  cur  <- rosters$season == s
  prev <- rosters[rosters$season == s - 1, ]
  j    <- match(rosters$p[cur], prev$p)
  stay <- !is.na(j) & runif(sum(cur)) < 0.75
  rosters$team[cur][stay] <- prev$team[j[stay]]
}
# mid-season trades: about 8% of players finish a season somewhere else
rosters <- rosters %>% mutate(team2 = ifelse(runif(n()) < 0.08,
                                             sample(N_TEAMS, n(), TRUE), team))

# ---- shots ------------------------------------------------------------------

SHOT_TYPES <- c("jumper", "layup", "floater", "dunk", "post", "tip", "heave")
type_probs <- list(
  HALFCOURT          = c(0.61, 0.23, 0.075, 0.040, 0.040, 0.002, 0.003),
  TRANSITION         = c(0.40, 0.38, 0.060, 0.135, 0.010, 0.010, 0.005),
  DIRECTSECONDCHANCE = c(0.27, 0.27, 0.030, 0.100, 0.050, 0.280, 0.000)
)
role_mult <- list(
  guard = c(1.15, 1.05, 1.4, 0.40, 0.3, 0.4, 1),
  wing  = c(1.05, 1.00, 1.0, 1.00, 0.7, 0.8, 1),
  big   = c(0.55, 0.95, 0.5, 2.40, 3.0, 2.6, 1)
)
MONTHS <- c("OCT", "NOV", "DEC", "JAN", "FEB", "MAR", "APR")
month_w <- c(23, 76, 70, 80, 59, 80, 37)

target_rate <- c(jumper = 0.378, layup = 0.535, floater = 0.417, dunk = 0.885,
                 post = 0.454, tip = 0.481, heave = 0.095)

simulate_season <- function(s, n) {
  ro  <- rosters[rosters$season == s, ]
  sh  <- sample(nrow(ro), n, TRUE, prob = players$usage[ro$p])
  mon <- sample(seq_along(MONTHS), n, TRUE, prob = month_w)
  # traded players shoot for their second team from February on
  off_team <- ifelse(mon >= 5, ro$team2[sh], ro$team[sh])
  def_team <- (off_team + sample(N_TEAMS - 1, n, TRUE) - 1) %% N_TEAMS + 1
  shooter  <- ro$p[sh]
  role     <- players$role[shooter]

  gamestate <- sample(names(type_probs), n, TRUE, prob = c(0.72, 0.205, 0.075))
  shottype  <- character(n)
  for (gs in names(type_probs)) for (r in names(role_mult)) {
    k <- which(gamestate == gs & role == r)
    if (length(k)) shottype[k] <- sample(SHOT_TYPES, length(k), TRUE,
                                         prob = type_probs[[gs]] * role_mult[[r]])
  }

  # distance (ft) and angle from the rim; 0 deg = straight on, 90 = baseline
  d <- numeric(n); ang <- runif(n, 0, 90)
  is <- function(t) shottype == t
  d[is("dunk")]    <- runif(sum(is("dunk")), 0.2, 3)
  d[is("tip")]     <- pmin(rgamma(sum(is("tip")), 2, 1.1), 6)
  d[is("layup")]   <- pmin(0.5 + rgamma(sum(is("layup")), 2, 0.9), 12)
  d[is("post")]    <- runif(sum(is("post")), 3, 14)
  d[is("floater")] <- runif(sum(is("floater")), 5, 15)
  d[is("heave")]   <- runif(sum(is("heave")), 36, 80)
  ang[is("heave")] <- runif(sum(is("heave")), 0, 25)
  j    <- which(is("jumper"))
  kind <- sample(c("mid", "corner3", "arc3", "deep"), length(j), TRUE, c(0.40, 0.11, 0.44, 0.05))
  d[j] <- ifelse(kind == "mid", runif(length(j), 8, 22.5),
          ifelse(kind == "corner3", runif(length(j), 22.05, 23.6),
          ifelse(kind == "arc3", 23.8 + rgamma(length(j), 2, 1.6), runif(length(j), 27, 34))))
  ang[j] <- ifelse(kind == "corner3", runif(length(j), 70, 89), ang[j])
  # keep shots inside the sidelines (|y| <= 25) and threes outside the arc
  ang <- pmin(ang, ifelse(d > 25, asin(pmin(25 / d, 1)) * 180 / pi, 90))
  locationx <- RIM_X + d * cos(ang * pi / 180)
  locationy <- d * sin(ang * pi / 180) * sample(c(-1, 1), n, TRUE)
  three <- d >= 23.75 | (abs(locationy) >= 22 & d >= 22)
  three[is("dunk") | is("tip")] <- FALSE
  # exactly four three-point layups, all in the first season
  lay3 <- if (s == 1) sample(which(is("layup")), 4) else integer(0)
  d[lay3] <- 23.9; ang[lay3] <- 30
  locationx[lay3] <- RIM_X + d[lay3] * cos(ang[lay3] * pi / 180)
  locationy[lay3] <- d[lay3] * sin(ang[lay3] * pi / 180)
  three[lay3] <- TRUE
  # layups and floaters can't be threes otherwise
  three[(is("layup") | is("floater") | is("post")) & !seq_len(n) %in% lay3] <- FALSE

  # dribbles: catch-and-shoot or off the bounce
  p0 <- c(jumper = 0.58, layup = 0.30, floater = 0.12, dunk = 0.42, post = 0.10,
          tip = 0.97, heave = 0.50)[shottype]
  dribbles <- ifelse(runif(n) < p0, 0L, 1L + rnbinom(n, size = 1.4, mu = 2.6))
  dribbles <- pmin(dribbles, 36L)

  shotclock <- ifelse(gamestate == "TRANSITION", 24 - rgamma(n, 2, 0.45),
               ifelse(gamestate == "DIRECTSECONDCHANCE", 14 - abs(rnorm(n, 0, 3)),
                      rnorm(n, 11, 5.2)))
  shotclock[is("heave")] <- runif(sum(is("heave")), 0, 1.5)
  shotclock <- round(pmin(pmax(shotclock, 0), 24), 2)

  speed <- rgamma(n, 1.6, 1.6 / 3.2)
  speed <- speed + ifelse(gamestate == "TRANSITION", 4, 0) +
    ifelse(shottype %in% c("dunk", "layup"), 2.5, 0) - ifelse(dribbles == 0 & is("jumper"), 1.2, 0)
  speed <- pmin(pmax(speed, 0), 24)

  # closest defender, contesters and the defender's last second of approach
  base_cd <- c(jumper = 4.9, layup = 3.0, floater = 3.6, dunk = 3.2, post = 2.4,
               tip = 2.6, heave = 9)[shottype]
  cdd <- rlnorm(n, log(base_cd) + ifelse(gamestate == "TRANSITION", 0.2, 0), 0.5)
  cdd <- pmin(pmax(cdd, 0.04), 60)
  contested <- cdd < 1.2 | runif(n) < plogis((7.5 - cdd) * 1.1)
  extra <- ifelse(contested, sample(0:3, n, TRUE, c(0.82, 0.17, 0.0095, 0.0005)), 0L)
  num_contesters <- as.integer(contested) + extra

  def_ro <- split(rosters$p[rosters$season == s], rosters$team[rosters$season == s])
  pick_def <- function(t) def_ro[[t]][sample(length(def_ro[[t]]), 1)]
  closestdef <- vapply(def_team, pick_def, 1L)
  cont_id <- matrix(NA_character_, n, 4); cont_d <- matrix(NA_real_, n, 4)
  cont_id[contested, 1] <- players$player_id[closestdef[contested]]
  cont_d[contested, 1]  <- cdd[contested]
  for (k in 2:4) {
    m <- which(num_contesters >= k)
    if (!length(m)) next
    others <- vapply(m, function(i) {
      pool <- setdiff(def_ro[[def_team[i]]], closestdef[i])
      pool[sample(length(pool), 1)]
    }, 1L)
    cont_id[m, k] <- players$player_id[others]
    cont_d[m, k]  <- cdd[m] + rgamma(length(m), 1.5, 1.5 / 1.6)
  }

  close_speed <- rnorm(n, 4, 4)                   # ft/s toward the shooter
  tt <- c(1, 0.75, 0.5, 0.25)
  appr <- sapply(tt, function(t) pmax(cdd + close_speed * t + rnorm(n, 0, 0.4), 0.05))
  approach <- sprintf("{%.10f,%.10f,%.10f,%.10f}", appr[, 1], appr[, 2], appr[, 3], appr[, 4])
  late_close <- pmax((appr[, 4] - cdd) / 0.25, 0)

  # ---- make probability ---------------------------------------------------
  pl <- players[shooter, ]
  df_skill <- players$def_skill[closestdef]
  shape <- ifelse(is("jumper"),
                  -0.045 * d - 0.8 * exp(-cdd / 2.5) + ifelse(three, pl$three_skill, pl$jump_skill) +
                    0.12 * (three & abs(locationy) >= 22) + 0.10 * (dribbles == 0) -
                    0.02 * speed - 0.05 * (dribbles >= 6),
           ifelse(is("layup"), -0.12 * d - 0.9 * exp(-cdd / 1.8) + pl$rim_skill,
           ifelse(is("tip"), -0.15 * d + pl$rim_skill,
           ifelse(is("dunk"), -0.3 * exp(-cdd / 1.5) + 0.5 * pl$rim_skill,
           ifelse(is("post"), -0.06 * d - 0.5 * exp(-cdd / 2) + 0.5 * (pl$rim_skill + pl$jump_skill),
           ifelse(is("floater"), -0.05 * d - 0.4 * exp(-cdd / 2.5) + 0.5 * pl$jump_skill + 0.3 * pl$rim_skill,
                  -0.02 * (d - 40)))))))
  shape <- shape - 0.15 * (shotclock < 4) - 0.012 * pmin(late_close, 25) -
    0.08 * pmax(num_contesters - 1, 0) + 0.08 * (gamestate == "TRANSITION") +
    0.05 * (gamestate == "DIRECTSECONDCHANCE") + df_skill +
    teams$off_skill[off_team] + teams$def_skill[def_team]

  tibble(
    season_id = seasons[s], off_team_id = teams$team_id[off_team],
    def_team_id = teams$team_id[def_team], shooter_id = pl$player_id,
    closestdef_id = players$player_id[closestdef], num_contesters = num_contesters,
    contester1_id = cont_id[, 1], contester2_id = cont_id[, 2],
    contester3_id = cont_id[, 3], contester4_id = cont_id[, 4],
    distcont1 = cont_d[, 1], distcont2 = cont_d[, 2],
    distcont3 = cont_d[, 3], distcont4 = cont_d[, 4],
    contested = contested, month = MONTHS[mon], gamestate = gamestate,
    shottype = shottype, distance = sqrt((locationx - RIM_X)^2 + locationy^2),
    locationx = round(locationx, 2), locationy = round(locationy, 2), three = three,
    dribblesbefore = dribbles, shotclock = shotclock, closestdefdist = cdd,
    closestdefapproach = approach, shooterspeed = speed,
    .shape = shape
  )
}

shots <- bind_rows(simulate_season(1, N_TRAIN / 2), simulate_season(2, N_TRAIN / 2),
                   simulate_season(3, N_TEST))

# per-type intercepts so each shot type's make rate lands near a realistic level
p <- numeric(nrow(shots))
for (t in SHOT_TYPES) {
  k <- which(shots$shottype == t)
  b <- uniroot(function(b) mean(plogis(b + shots$.shape[k])) - target_rate[[t]], c(-10, 10))$root
  p[k] <- plogis(b + shots$.shape[k])
}
shots$outcome <- runif(nrow(shots)) < p

ids <- sample(nrow(shots))
shots$shot_id <- ids
train <- shots[seq_len(N_TRAIN), ] %>% arrange(shot_id)
test  <- shots[-seq_len(N_TRAIN), ] %>% arrange(shot_id)

cols <- c("shot_id", "outcome", "season_id", "off_team_id", "def_team_id", "shooter_id",
          "closestdef_id", "num_contesters", "contester1_id", "contester2_id",
          "contester3_id", "contester4_id", "distcont1", "distcont2", "distcont3",
          "distcont4", "contested", "month", "gamestate", "shottype", "distance",
          "locationx", "locationy", "three", "dribblesbefore", "shotclock",
          "closestdefdist", "closestdefapproach", "shooterspeed")

dir.create("data", showWarnings = FALSE)
write_csv(train[, cols], "data/training.csv.gz")
write_csv(test[, setdiff(cols, "outcome")], "data/testing.csv.gz")
cat("wrote", nrow(train), "training and", nrow(test), "testing shots | make rate:",
    round(mean(train$outcome), 4), "\n")
