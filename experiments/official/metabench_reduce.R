# metabench (Kipnis et al., 2025) -- the official analysis/reduce.R pipeline driven on
# one ATDrive protocol cell.  Nothing algorithmic is re-typed here: the function
# definitions of analysis/utils.R, analysis/reduce.R and ATLAS scripts/04_pirt_accuracy.r
# are evaluated verbatim out of the read-only clones by `source.functions` below, and this
# file only supplies the arguments those functions expect.  See the ADAPTATIONS section of
# experiments/official/metabench.py for every place where that plumbing deviates.
#
# Line numbers in the comments below refer to the pinned clones:
#   metabench  adkipnis/metabench       abfeb64cd8e6da1b753c8d133ba034f7ed168afb
#   ATLAS      Peiyu-Georgia-Li/ATLAS   c0bc19b33aaa11703b8258377abb367533b8a824
#
#   Rscript metabench_reduce.R fit      <workdir>
#   Rscript metabench_reduce.R estimate <workdir>

MB_ROOT <- "/data2/jeongtae/official_baselines/metabench/analysis"
ATLAS_PIRT <- "/data2/jeongtae/official_baselines/ATLAS/scripts/04_pirt_accuracy.r"

# Evaluate exactly the top-level `name <- function(...)` definitions of an official
# script, skipping its side effects (reduce.R and 04_pirt_accuracy.r are scripts: they
# load benchmark .rds files, parse argv and write plots at the top level).
source.functions <- function(path, env = globalenv()) {
  got <- character(0)
  for (e in parse(path)) {
    if (is.call(e) && length(e) == 3 && as.character(e[[1]]) %in% c("<-", "=") &&
        is.call(e[[3]]) && identical(as.character(e[[3]][[1]]), "function")) {
      eval(e, envir = env)
      got <- c(got, as.character(e[[2]]))
    }
  }
  got
}
loaded <- c(source.functions(file.path(MB_ROOT, "utils.R")),      # run.mirt, get.theta
            source.functions(file.path(MB_ROOT, "reduce.R")),     # collect.item.info,
            source.functions(ATLAS_PIRT))                         # get.info.quantiles,
for (f in c("run.mirt", "get.theta", "collect.item.info", "get.info.quantiles",
            "select.items", "summarize.info", "make.df.score", "compute_3pl_prob", "compute_pirt_accuracy")) {
  if (!exists(f)) stop(sprintf("official function %s did not load", f))
}

args <- commandArgs(trailingOnly = TRUE)
phase <- args[1]
wd <- args[2]
cfg <- jsonlite::fromJSON(file.path(wd, "config.json"))
set.seed(cfg$seed)
warn.log <- character(0)
catch <- function(expr, tag) withCallingHandlers(expr, warning = function(w) {
  warn.log <<- c(warn.log, paste0(tag, ": ", conditionMessage(w))); invokeRestart("muffleWarning")
})

# ---------------------------------------------------------------- fit -------
if (phase == "fit") {
  data0 <- read.csv(file.path(wd, "cal.csv"), check.names = FALSE)
  scores <- read.csv(file.path(wd, "scores.csv"))$score

  # metabench item pre-selection step 0 (analysis/preprocess.R:154 `items$sd <-
  # apply(data, 2, sd)` and :162-164 "item answers must vary", sd <= 0.01); mirt refuses
  # single-category items outright.  DEVIATION (ADAPTATION 5): na.rm = TRUE and the
  # !is.na(sds) guard are additions -- their matrix has no NA cells, ours does
  # (ADAPTATION 9), and their bare sd() would return NA and fail the comparison.  The
  # guard also drops routes with no record at all from the eligible pool.
  sds <- apply(data0, 2, function(x) stats::sd(x, na.rm = TRUE))
  keep <- !is.na(sds) & sds > 0.01
  data <- data0[, keep, drop = FALSE]
  dropped <- colnames(data0)[!keep]

  model <- catch(run.mirt(data, 1, cfg$model_type, tol = cfg$tol, ncycles = cfg$ncycles), "run.mirt")
  theta <- catch(get.theta(model, cfg$theta_type), "get.theta")
  cf <- mirt::coef(model, simplify = TRUE, rotate = "none")$items

  # theta grid, hyperparams$grid.type == 2 (reduce.R:205-210); ADAPTATION 4
  theta.range <- range(theta[, 1])
  theta.grid <- seq(theta.range[1], theta.range[2], length.out = cfg$n_quant)
  info.items <- collect.item.info(model, theta.grid, colnames(data))
  # reduce.R:219 `items <- merge(items, summarize.info(info.items), by="item")`
  items <- merge(data.frame(item = colnames(data), stringsAsFactors = FALSE),
                 summarize.info(info.items), by = "item")

  out <- list(); subs <- list()
  for (B in cfg$budgets) {
    key <- as.character(B)
    info.quantiles <- get.info.quantiles(info.items, theta.grid, steps = B)
    sel <- catch(select.items(items, info.items, info.quantiles, threshold = cfg$threshold), "select.items")
    sel.names <- as.character(sel$item)
    data.sub <- data[, sel.names, drop = FALSE]                       # create.subtest
    model.sub <- catch(run.mirt(data.sub, 1, cfg$model_type, tol = cfg$tol, ncycles = cfg$ncycles),
                       paste0("run.mirt.sub.B", B))
    theta.sub <- catch(get.theta(model.sub, cfg$theta_type, data.sub), paste0("get.theta.sub.B", B))

    # score readout: reduce.R:105 `mgcv::gam(score ~ s(theta, bs = "ad"), data = df.train)`
    # on their own df (reduce.R:94 make.df.score).  The error branch below is a
    # WRAPPER-DEFINED fallback, not metabench code, and has never triggered: on 4-12
    # calibration planners the adaptive smooth only warns (ADAPTATION 6).
    df.train <- make.df.score(scores, theta.sub)
    readout <- catch(tryCatch({ m <- mgcv::gam(score ~ s(theta, bs = "ad"), data = df.train)
                                list(mod = m, kind = "gam") },
                              error = function(e) {
                                warn.log <<- c(warn.log, paste0("gam.B", B, " ERROR: ", conditionMessage(e)))
                                list(mod = stats::lm(score ~ theta, data = df.train), kind = "wrapper.lin") }),
                     paste0("readout.B", B))
    # WRAPPER-DEFINED linear readout (theta -> score), reported as a variant.  This is NOT
    # metabench's linear baseline: theirs is meta.R:284 `lm(grand ~ ., data = data.train)`,
    # the grand score on the ITEM RESPONSES, which is rank-deficient with 4-12 respondents
    # and 30-165 regressors and is therefore not run here.  No `lm(score ~ theta)` exists
    # in the metabench repo.
    wrapper.lin <- stats::lm(score ~ theta, data = df.train)
    subs[[key]] <- list(model.sub = model.sub, readout = readout$mod, wrapper.lin = wrapper.lin, sel = sel.names)
    out[[key]] <- list(items = sel.names, n_items = length(sel.names), readout = readout$kind,
                       theta_sub_cal = as.numeric(theta.sub[, 1]))
  }

  saveRDS(list(model = model, subs = subs, fitted.items = colnames(data)), file.path(wd, "fit.rds"))
  jsonlite::write_json(list(
    budgets = out,
    info = list(model_type = cfg$model_type, theta_type = cfg$theta_type, grid_type = 2L,
                n_quant = cfg$n_quant, threshold = cfg$threshold, ncycles = cfg$ncycles, tol = cfg$tol,
                n_bank = ncol(data0), n_fitted = ncol(data), dropped_low_variance = dropped,
                cal_theta = as.numeric(theta[, 1]), cal_score = scores,
                a_min = min(cf[, "a1"]), a_max = max(cf[, "a1"]), a_mean = mean(cf[, "a1"]),
                n_a_nonpositive = sum(cf[, "a1"] <= 0),          # ADAPTATION 7(b)
                b_mean = mean(-cf[, "d"] / cf[, "a1"]), b_sd = stats::sd(-cf[, "d"] / cf[, "a1"]),
                warnings = unique(warn.log))),
    file.path(wd, "fit.json"), auto_unbox = TRUE, digits = 12, null = "null")

# ------------------------------------------------------------ estimate ------
} else if (phase == "estimate") {
  fit <- readRDS(file.path(wd, "fit.rds"))
  resp <- jsonlite::fromJSON(file.path(wd, "resp.json"), simplifyVector = FALSE)
  fitted.items <- fit$fitted.items
  bank.items <- cfg$bank_items

  # full-fit item parameters for every bank route, in the (X, a1, d, g) layout that
  # ATLAS's prepare_item_parameters (04_pirt_accuracy.r:34) reads; routes dropped by the
  # variance filter get a1 = d = NA and g = 0.5, so their compute_3pl_prob
  # (04_pirt_accuracy.r:63-65) returns its guessing fallback.  Fitted routes get g = 0, so
  # the SAME `a <= 0` branch returns p = 0 for an unobserved route with a non-positive
  # discrimination -- see ADAPTATION 7(b); the per-budget counts are reported below.
  cf <- mirt::coef(fit$model, simplify = TRUE, rotate = "none")$items
  ip <- data.frame(X = bank.items, a1 = NA_real_, d = NA_real_, g = 0.5, stringsAsFactors = FALSE)
  m <- match(fitted.items, bank.items)
  ip$a1[m] <- cf[, "a1"]; ip$d[m] <- cf[, "d"]; ip$g[m] <- 0
  utils::write.csv(ip, file.path(wd, "item_params.csv"), row.names = FALSE)
  item_params <- prepare_item_parameters(file.path(wd, "item_params.csv"))

  a.na <- is.na(item_params$a)
  a.nonpos <- !a.na & item_params$a <= 0

  out <- list()
  for (key in names(resp)) {
    sub <- fit$subs[[key]]
    yv <- as.numeric(unlist(resp[[key]]))                       # in sub$sel order
    stopifnot(length(yv) == length(sub$sel))

    # metabench step 3 for a new subject: theta on the refit subtest
    rs <- as.data.frame(matrix(yv, nrow = 1)); colnames(rs) <- sub$sel
    theta.new <- catch(get.theta(sub$model.sub, cfg$theta_type, rs), paste0("get.theta.new.B", key))
    est <- as.numeric(stats::predict(sub$readout, newdata = data.frame(theta = theta.new[1, 1])))

    # theta on the FULL fit restricted to the selected routes (scale of item_params)
    full.pat <- rep(NA_real_, length(fitted.items))
    full.pat[match(sub$sel, fitted.items)] <- yv
    rf <- as.data.frame(matrix(full.pat, nrow = 1)); colnames(rf) <- fitted.items
    theta.full <- catch(get.theta(fit$model, cfg$theta_type, rf), paste0("get.theta.full.B", key))

    pirt <- function(th) {
      si <- data.frame(item_id = sub$sel, score = yv, theta = th, stringsAsFactors = FALSE)
      f <- file.path(wd, paste0("selected_", key, ".csv")); utils::write.csv(si, f, row.names = FALSE)
      compute_pirt_accuracy("eval_planner", item_params, NULL, f)$pirt_accuracy
    }
    # ADAPTATION 7: which imputation branch each unobserved bank route falls into.
    unobs <- !(item_params$item_id %in% sub$sel)
    out[[key]] <- list(est = est, theta_sub = theta.new[1, 1], theta_full = theta.full[1, 1],
                       sample_mean = mean(yv), pirt = pirt(theta.full[1, 1]),
                       pirt_refit = pirt(theta.new[1, 1]),
                       pirt_p_zero = sum(unobs & a.nonpos), pirt_p_half = sum(unobs & a.na),
                       wrapper_lin = as.numeric(stats::predict(
                         sub$wrapper.lin, newdata = data.frame(theta = theta.new[1, 1]))))
  }
  jsonlite::write_json(list(budgets = out, warnings = unique(warn.log)),
                       file.path(wd, "est.json"), auto_unbox = TRUE, digits = 12, null = "null")
} else {
  stop("unknown phase")
}
