#!/usr/bin/env Rscript
# ATLAS (Li et al. 2025) on the ATDrive protocol, through the OFFICIAL code in
# /data2/jeongtae/official_baselines/ATLAS/scripts (read-only, never edited).
#
# The algorithmic core is NOT re-implemented here: the function definitions of
# 03_atlas_cat.r (prepare_item_bank, run_atlas) and of 04_pirt_accuracy.r
# (prepare_item_parameters, compute_3pl_prob, compute_pirt_accuracy) are parsed
# out of the official files and evaluated verbatim (source_functions below); the
# mirt fit is the literal call of 01_fit_irt.r:96-97. Only the file / argument
# plumbing around them lives here, plus the one function that had to be replaced
# (score_response, 03:120-137 -> the stdout/stdin response oracle, so that the
# evaluation planner's outcomes are never handed to R in bulk).
# See the ADAPTATIONS section of atlas.py for every deviation.
#
#   Rscript atlas_cat.R fit <calib.csv> <workdir>
#       calib.csv : rows = calibration planners, columns X<bank index>, cells 0/1/NA
#       writes <workdir>/irt_item_parameters_combined.csv (the file 03/04 read)
#              <workdir>/fit_info.json
#   Rscript atlas_cat.R cat <workdir> <seed> <configs.json> <out.json>
#       configs.json : [{name, max_items, min_items, se_theta_stop}, ...]; each config
#               is one run_atlas call preceded by set.seed(seed)
#       responses  : for every administered item R writes "@ASK <item id>" on stdout
#               and reads the 0/1 outcome back on stdin (the caller is the only
#               holder of the evaluation planner's outcome vector)

suppressPackageStartupMessages({ library(mirt); library(catR); library(jsonlite) })

OFFICIAL <- "/data2/jeongtae/official_baselines/ATLAS/scripts"

# Evaluate only the `name <- function(...)` top-level expressions of an official
# script (their scripts run a main section on source(), with hard-coded paths).
source_functions <- function(path, envir = globalenv()) {
  got <- character(0)
  for (e in parse(path, keep.source = FALSE)) {
    if (is.call(e) && identical(e[[1]], as.name("<-")) && is.call(e[[3]]) &&
        identical(e[[3]][[1]], as.name("function"))) {
      eval(e, envir)
      got <- c(got, as.character(e[[2]]))
    }
  }
  got
}

args <- commandArgs(trailingOnly = TRUE)
mode <- args[1]

# ── fit: 01_fit_irt.r on the calibration matrix (single chunk) ───────────────
if (mode == "fit") {
  calib_csv <- args[2]; workdir <- args[3]
  data <- read.csv(calib_csv)                     # columns X<bank idx>, as read.csv names them
  n_bank <- ncol(data)
  # 01:75-76 `na.omit(data)` drops every planner that has any missing record (2 of 4
  # at K_cal = 4, 5 of 12 at K_cal = 12 on average); mirt treats NA cells as missing,
  # so NA is kept (ADAPTATION 2).
  # 01:78-85 constant-column / constant-row cleaning, evaluated on the non-NA
  # entries (with NA present their length(unique(x)) == 1 would never fire).
  constant_cols <- apply(data, 2, function(x) length(unique(na.omit(x))) <= 1)
  clean_data    <- data[, !constant_cols, drop = FALSE]
  cat("Dropped", sum(constant_cols), "constant columns.\n")
  constant_rows <- apply(clean_data, 1, function(x) length(unique(na.omit(x))) <= 1)
  clean_data    <- clean_data[!constant_rows, , drop = FALSE]
  cat("Dropped", sum(constant_rows), "constant rows.\n")
  cat("Dimensions of cleaned data:", dim(clean_data), "\n")
  dat <- clean_data                               # one chunk = the whole bank (ADAPTATION 1)

  warns <- character(0); msgs <- character(0)
  t0 <- Sys.time()
  model <- withCallingHandlers(
    mirt(dat, 1, itemtype = "3PL", method = "EM",   # 01_fit_irt.r:96-97 verbatim
         technical = list(NCYCLES = 100000), verbose = FALSE),
    warning = function(w) { warns <<- c(warns, conditionMessage(w)); invokeRestart("muffleWarning") },
    message = function(m) { msgs <<- c(msgs, conditionMessage(m)); invokeRestart("muffleMessage") })
  fit_s <- as.numeric(Sys.time() - t0, units = "secs")
  theta_cal <- tryCatch(fscores(model, method = "EAP", full.scores = TRUE,       # 01:101-102
                                full.scores.SE = TRUE, quadpts = 61),
                        error = function(e) NULL)
  item_params <- coef(model, simplify = TRUE)$items                              # 01:103
  # 01:111 writes irt_item_parameters_<chunk>.csv with row.names; the pipeline
  # rbinds the chunks into irt_item_parameters_combined.csv, the name 03/04 read
  # (no linking needed for a single chunk).
  write.csv(item_params, file.path(workdir, "irt_item_parameters_combined.csv"), row.names = TRUE)

  a1 <- item_params[, "a1"]; d <- item_params[, "d"]; g <- item_params[, "g"]
  b  <- -d / a1
  drop_idx <- as.integer(sub("^X", "", colnames(data)[constant_cols]))
  drop_val <- as.numeric(apply(data[, constant_cols, drop = FALSE], 2,
                               function(x) { v <- unique(na.omit(x)); if (length(v) == 1) v[1] else NA }))
  info <- list(
    n_bank = n_bank, n_items_fit = ncol(dat), n_rows_fit = nrow(dat), n_rows_in = nrow(data),
    dropped_constant_items = drop_idx, dropped_constant_values = drop_val,
    dropped_constant_rows = sum(constant_rows),
    item_ids = as.integer(sub("^X", "", rownames(item_params))),
    converged = isTRUE(extract.mirt(model, "converged")),
    em_iterations = extract.mirt(model, "iterations"),
    logLik = extract.mirt(model, "logLik"),
    fit_seconds = fit_s,
    warnings = warns, messages = msgs,
    a1 = list(min = min(a1), median = median(a1), max = max(a1), n_nonpositive = sum(a1 <= 0)),
    b  = list(min = min(b[is.finite(b)]), median = median(b[is.finite(b)]), max = max(b[is.finite(b)]),
              n_nonfinite = sum(!is.finite(b)), n_abs_gt_5 = sum(abs(b) > 5, na.rm = TRUE)),
    g  = list(min = min(g), median = median(g), max = max(g)),
    theta_cal_eap = if (is.null(theta_cal)) NULL else as.numeric(theta_cal[, "F1"]),
    theta_cal_se  = if (is.null(theta_cal)) NULL else as.numeric(theta_cal[, "SE_F1"]))
  writeLines(toJSON(info, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), file.path(workdir, "fit_info.json"))
  cat("fit done in", round(fit_s, 1), "s; converged:", info$converged, "\n")
  quit(status = 0)
}

# ── cat: 03_atlas_cat.r run_atlas + 04_pirt_accuracy.r compute_pirt_accuracy ─
if (mode == "cat") {
  workdir <- args[2]; seed <- as.integer(args[3])
  configs <- fromJSON(args[4], simplifyVector = FALSE); out_json <- args[5]

  f03 <- source_functions(file.path(OFFICIAL, "03_atlas_cat.r"))
  f04 <- source_functions(file.path(OFFICIAL, "04_pirt_accuracy.r"))
  stopifnot(all(c("prepare_item_bank", "score_response", "run_atlas") %in% f03),
            all(c("prepare_item_parameters", "compute_3pl_prob", "compute_pirt_accuracy") %in% f04))

  # ADAPTATION 4: their score_response (03:120-137) indexes a response matrix that
  # holds every model's outcome on every item; here the outcomes live in the caller,
  # which serves one item at a time.  Same signature, same return value.
  stdin_con <- file("stdin", open = "rt")
  score_response <- function(model_name, item_id, response_data, response_model_names) {
    cat("@ASK", item_id, "\n"); flush(stdout())
    as.numeric(readLines(stdin_con, n = 1L))
  }

  # ADAPTATION 5: catR 3.17's nextItem ends with set.seed(NULL) (line 370 of 372 of
  # its body), which reseeds R's RNG from the clock, so a run_atlas trajectory is not
  # reproducible under set.seed and two budgets do not share a prefix.  The shadow
  # below seeds the stream deterministically from the cell seed and the call index and
  # then delegates to catR::nextItem with the arguments unchanged; the MFI rule and the
  # randomesque draw are theirs.
  .atlas_seed <- 0L
  .atlas_call <- 0L
  nextItem <- function(...) {
    .atlas_call <<- .atlas_call + 1L
    set.seed(as.integer((as.numeric(.atlas_seed) * 1000 + .atlas_call) %% 2147483647))
    catR::nextItem(...)
  }

  params_file <- file.path(workdir, "irt_item_parameters_combined.csv")
  item_data   <- prepare_item_bank(params_file)          # 03:63-113 (a, b = -d/a1, c = g, d = u)
  item_params <- prepare_item_parameters(params_file)    # 04:34-56
  model_name  <- "eval"                                  # the single evaluated planner
  response_data <- NULL                                  # only passed through to score_response

  out <- list()
  for (cfg in configs) {
    DIRECTORY <<- paste0(workdir, "/", cfg$name, "/")   # 03:61 global that run_atlas writes into
    t0 <- Sys.time()
    .atlas_seed <- seed; .atlas_call <- 0L               # ADAPTATION 5 (shadow nextItem)
    set.seed(seed)                                       # 03:367 (theirs: 123 + model index)
    res <- run_atlas(item_bank = item_data$item_bank, item_info = item_data$item_info,
                     model_name = model_name, response_data = response_data,
                     response_model_names = model_name,
                     start_theta = 0, min_items = cfg$min_items, max_items = cfg$max_items,
                     se_theta_stop = cfg$se_theta_stop, verbose = FALSE)    # 03:369-380
    items_file <- paste0(DIRECTORY, "selected_items_", cfg$se_theta_stop, "/", model_name, "_items.csv")
    pirt <- compute_pirt_accuracy(model_name, item_params, response_data, items_file)   # 04:76-140
    out[[cfg$name]] <- list(
      items = as.integer(sub("^X", "", res$results$item_id)),
      scores = res$results$score, thetas = res$results$theta, ses = res$results$se,
      final_theta = res$final_theta, final_se = res$final_se, num_items = res$num_items,
      pirt_accuracy = pirt$pirt_accuracy, avg_observed = pirt$avg_observed,
      avg_predicted = pirt$avg_predicted, n_subset_items = pirt$n_subset_items,
      n_all_items = pirt$n_all_items,
      seconds = as.numeric(Sys.time() - t0, units = "secs"))
  }
  writeLines(toJSON(out, auto_unbox = TRUE, digits = NA, null = "null", na = "null"), out_json)
  cat("@DONE\n"); flush(stdout())
  quit(status = 0)
}

stop("unknown mode: ", mode)
