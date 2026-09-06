#!/usr/bin/env Rscript
# The two classical static information orders through catR 3.17 (Magis & Raiche,
# JSS 2012 -- the reference CAT implementation) on a mirt 1.47 2PL fit.
# Nothing algorithmic is re-implemented here: the item calibration is mirt::mirt,
# the item information is catR::Ii (Birnbaum 1968), the prior-weighted
# information is catR::MWI(type = "MPWI") (van der Linden 1998), the ability is
# catR::thetaEst(method = "BM") and the response probabilities of the
# unadministered items are catR::Pi.  Only the file / argument plumbing is here.
# Every deviation is listed in the ADAPTATIONS section of catr.py.
#
#   Rscript catr_static.R fit <calib.csv> <workdir>
#       calib.csv : rows = calibration planners, columns X<bank index> in bank
#                   order, cells 0 / 1 / NA (no record)
#       writes <workdir>/catr_fit.json  (itemBank, both information vectors,
#              both orders as 0-based bank indices, fit diagnostics)
#
#   Rscript catr_static.R est <payload.json> <out.json>
#       payload.json : {itemBank: [[a, b, c, d], ...],
#                       admin: [{name, items (0-based, administration order),
#                                x (0/1)}, ...]}
#       for each entry: thetaEst(BM, norm(0,1)) on the administered items and the
#       p-IRT plug-in (sum of the observed responses + sum of catR::Pi over the
#       unadministered items) / n_bank

suppressPackageStartupMessages({ library(mirt); library(catR); library(jsonlite) })

args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]

# --- catR item bank from a mirt 2PL fit -------------------------------------
# catR's dichotomous itemBank is the 4PL matrix (a, b, c, d) -- ?MWI Details:
# "itemBank must be a matrix with one row per item and four columns, with the
# values of the discrimination, the difficulty, the pseudo-guessing and the
# inattention parameters (in this order)".  mirt's 2PL slope/intercept
# parameterisation a1 * theta + d is turned into the difficulty b = -d / a1
# (the same conversion the official ATLAS 03_atlas_cat.r:prepare_item_bank
# applies to a mirt fit), c = 0 and d = 1 give the 2PL.
bank_from_mirt <- function(item_params) {
  a1 <- item_params[, "a1"]; dd <- item_params[, "d"]
  cbind(a = as.numeric(a1), b = as.numeric(-dd / a1), c = 0, d = 1)
}

# --- order, ties broken by the lowest bank index ----------------------------
# order(-info, seq_along(info)) sorts by decreasing information and, within an
# exact tie, by increasing bank index.  (R's order() is not documented to be
# stable for the default method, so the index is passed as an explicit second
# key rather than relied upon.)  catR's own nextItem would break the same tie by
# sample() among the tied group, so this is a DEVIATION from catR's convention,
# and it is load-bearing at K_cal = 4, where the budget boundary falls inside a
# tie group: it moves the estimate and it happens to favour this baseline at
# B = 30.  The measured effect is in the tie-rule paragraph of catr.py.
info_order <- function(info) order(-info, seq_along(info))

# --- empirical-Bayes selection of the intercept-prior sd (ADAPTATION 1) ------
# Grid = atdrive/calibration.py:20 SIGMA_B_GRID, applied on mirt's intercept
# scale.  Criterion = atdrive/curves.py:item_marginal_loglik, i.e. for the
# (a_hat, theta_hat) of the fit at that sd,
#     sum_j log int prod_k P(y_kj | a_j, d) N(d; 0, sd^2) dd
# on a 0.025-spaced grid (the spacing of ATDrive's own BG grid).  The grid RANGE
# is [-12, 12] rather than BG's [-10, 10] (the intercept d is on a wider scale
# than b when a < 1); on a matched synthetic case (a == 1, so d = -b) the two
# criteria agree to 6 decimals for sd <= 1.5 and differ by 1.7e-5 / 1.5e-2 nats
# at sd = 2 / 3 out of ~160, so the range is immaterial to the argmax.
SIGMA_D_GRID <- c(0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
DG <- seq(-12, 12, by = 0.025)

eb_marginal_loglik <- function(Y, MK, a, theta, sd) {
  lprior <- -0.5 * (DG / sd)^2 - log(sd * sqrt(2 * pi)) + log(DG[2] - DG[1])
  tot <- 0
  for (j in seq_len(ncol(Y))) {
    k <- which(MK[, j])
    if (!length(k)) next
    Z <- outer(a[j] * theta[k], DG, "+")                       # |k| x length(DG)
    ll <- colSums(Y[k, j] * plogis(Z, log.p = TRUE) +
                  (1 - Y[k, j]) * plogis(-Z, log.p = TRUE)) + lprior
    m <- max(ll)
    tot <- tot + m + log(sum(exp(ll - m)))
  }
  tot
}

# --------------------------------------------------------------------------- #
if (mode == "fit") {
  calib_csv <- args[2]; workdir <- args[3]
  seed <- if (length(args) >= 4) as.integer(args[4]) else 0L
  dat <- read.csv(calib_csv)
  J <- ncol(dat); K <- nrow(dat)
  bank_idx <- as.integer(sub("^X", "", colnames(dat)))
  stopifnot(identical(bank_idx, 0:(J - 1L)))

  n_cat <- apply(dat, 2, function(x) length(unique(na.omit(x))))
  const_items <- which(n_cat <= 1L)
  if (any(n_cat < 1L)) stop("bank item(s) with no calibration record at all")

  # PRIOR through mirt's own mechanism (?mirt.model, "PRIOR = (1-10, a1, lnorm,
  # .2, .2)"): without it the all-pass / all-fail routes are not estimable.
  # ADAPTATION 1 / 2 in catr.py.  The intercept-prior sd is NOT fixed: it is
  # chosen per cell by empirical Bayes over ATDrive's own SIGMA_B_GRID, the same
  # rule and the same grid ATDrive's calibration uses for its sigma_b
  # (atdrive/calibration.py:65-78 _eb_fit).
  fit_one <- function(sd_d) {
    mod <- mirt.model(sprintf("F = 1-%d\nPRIOR = (1-%d, a1, lnorm, 0, 0.5), (1-%d, d, norm, 0, %s)",
                              J, J, J, format(sd_d, scientific = FALSE)))
    warns <- character(0); msgs <- character(0)
    set.seed(seed)                                  # mirt's EM is deterministic; kept for reproducibility
    t0 <- Sys.time()
    model <- withCallingHandlers(
      mirt(dat, mod, itemtype = "2PL", method = "EM",
           technical = list(customK = rep(2L, J)), verbose = FALSE),
      warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
      message = function(m) { msgs  <<- c(msgs,  conditionMessage(m)); invokeRestart("muffleMessage") })
    ip <- coef(model, simplify = TRUE)$items
    th <- as.numeric(fscores(model, method = "EAP", full.scores = TRUE, quadpts = 61)[, 1])
    list(model = model, item_params = ip, theta = th, warns = warns, msgs = msgs,
         secs = as.numeric(Sys.time() - t0, units = "secs"))
  }

  Ymat <- as.matrix(dat); MK <- !is.na(Ymat); Ymat[!MK] <- 0
  fits <- lapply(SIGMA_D_GRID, fit_one)
  eb <- vapply(seq_along(SIGMA_D_GRID), function(i)
    eb_marginal_loglik(Ymat, MK, as.numeric(fits[[i]]$item_params[, "a1"]),
                       fits[[i]]$theta, SIGMA_D_GRID[i]), numeric(1))
  best <- which.max(eb)
  sd_d <- SIGMA_D_GRID[best]
  fit <- fits[[best]]
  model <- fit$model; warns <- fit$warns; msgs <- fit$msgs
  fit_s <- sum(vapply(fits, function(f) f$secs, numeric(1)))

  item_params <- fit$item_params
  itemBank <- bank_from_mirt(item_params)
  if (any(!is.finite(itemBank)) || any(itemBank[, "a"] <= 0))
    stop("mirt returned a non-finite or non-positive-slope item")

  # EAP abilities of the calibration planners (mirt::fscores)
  theta_cal <- fit$theta

  # total_fisher: sum over the calibration planners' EAP abilities of catR::Ii
  t1 <- Sys.time()
  I_by_theta <- sapply(theta_cal, function(th) Ii(th, itemBank)$Ii)   # J x K
  total_info <- as.numeric(rowSums(matrix(I_by_theta, nrow = J)))
  total_s <- as.numeric(Sys.time() - t1, units = "secs")

  # marginal_fisher: catR::MWI step 0 (no administered item) with type = "MPWI"
  # -> integral of Ii(theta) * dnorm(theta, 0, 1) (van der Linden 1998 MPWI).
  t2 <- Sys.time()
  marginal_info <- vapply(seq_len(J), function(i)
    MWI(itemBank, i, x = NULL, it.given = NULL, type = "MPWI",
        priorDist = "norm", priorPar = c(0, 1)), numeric(1))
  marginal_s <- as.numeric(Sys.time() - t2, units = "secs")

  ord_total <- info_order(total_info)
  ord_marg  <- info_order(marginal_info)

  out <- list(
    n_bank = J, K_cal = K,
    itemBank = itemBank,
    theta_cal_eap = theta_cal,
    total_info = total_info, marginal_info = marginal_info,
    order_total_fisher = as.integer(ord_total - 1L),      # 0-based bank indices
    order_marginal_fisher = as.integer(ord_marg - 1L),
    info = list(
      n_bank = J, K_cal = K, n_responses = sum(!is.na(dat)),
      n_constant_items = length(const_items),
      constant_items = as.integer(const_items - 1L),
      prior = sprintf("a1 ~ lnorm(0, 0.5); d ~ norm(0, %s) [empirical Bayes over %s]",
                      format(sd_d, scientific = FALSE),
                      paste(format(SIGMA_D_GRID, scientific = FALSE), collapse = "/")),
      prior_sd_d = sd_d, prior_grid = SIGMA_D_GRID, prior_eb_loglik = eb,
      prior_sd_b_induced_min = sd_d / max(itemBank[, "a"]),
      prior_sd_b_induced_max = sd_d / min(itemBank[, "a"]),
      customK = TRUE, seed = seed,
      converged = isTRUE(extract.mirt(model, "converged")),
      em_iterations = extract.mirt(model, "iterations"),
      logLik = extract.mirt(model, "logLik"),
      n_prior_fits = length(SIGMA_D_GRID),          # fit_seconds is the sum over the grid
      fit_seconds = fit_s, total_info_seconds = total_s, marginal_info_seconds = marginal_s,
      a_min = min(itemBank[, "a"]), a_median = median(itemBank[, "a"]), a_max = max(itemBank[, "a"]),
      b_min = min(itemBank[, "b"]), b_median = median(itemBank[, "b"]), b_max = max(itemBank[, "b"]),
      theta_cal_min = min(theta_cal), theta_cal_max = max(theta_cal),
      n_ties_total = J - length(unique(total_info)),
      n_ties_marginal = J - length(unique(marginal_info)),
      n_warnings = length(warns), n_messages = length(msgs),
      warnings = unique(warns), catR_version = as.character(packageVersion("catR")),
      mirt_version = as.character(packageVersion("mirt"))))
  writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"),
             file.path(workdir, "catr_fit.json"))
  quit(status = 0)
}

# --------------------------------------------------------------------------- #
if (mode == "est") {
  payload_json <- args[2]; out_json <- args[3]
  payload <- fromJSON(payload_json, simplifyVector = FALSE)
  itemBank <- matrix(as.numeric(unlist(lapply(payload$itemBank, unlist))),
                     ncol = 4, byrow = TRUE,
                     dimnames = list(NULL, c("a", "b", "c", "d")))
  J <- nrow(itemBank)
  admin <- payload$admin

  out <- list()
  for (a in admin) {
    items <- as.integer(unlist(a$items)) + 1L        # to 1-based rows
    x <- as.numeric(unlist(a$x))
    stopifnot(length(items) == length(x), !anyDuplicated(items))
    it.given <- itemBank[items, , drop = FALSE]
    theta <- thetaEst(it.given, x, method = "BM", priorDist = "norm", priorPar = c(0, 1))
    rest <- setdiff(seq_len(J), items)
    P <- if (length(rest)) Pi(theta, itemBank[rest, , drop = FALSE])$Pi else numeric(0)
    out[[a$name]] <- list(
      theta = theta,
      se = semTheta(theta, it.given, x = x, method = "BM", priorDist = "norm", priorPar = c(0, 1)),
      pirt = (sum(x) + sum(P)) / J,
      sample_mean = mean(x),
      sum_P_rest = sum(P), n_rest = length(rest), n_items = length(items))
  }
  writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null"), out_json)
  quit(status = 0)
}

stop("unknown mode: ", mode)
