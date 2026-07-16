args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: extract_m1_tourism.R M1.rda tourism.rda output_dir")
m1_path <- args[[1]]
tourism_path <- args[[2]]
out_dir <- args[[3]]
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

extract_comp <- function(obj, benchmark) {
  if (!is.list(obj)) stop(sprintf("%s object is not a list", benchmark))
  value_rows <- vector("list", length(obj))
  meta_rows <- vector("list", length(obj))
  for (i in seq_along(obj)) {
    item <- obj[[i]]
    if (is.null(item$x) || is.null(item$xx)) stop(sprintf("%s item %d lacks x/xx", benchmark, i))
    history <- as.numeric(item$x)
    future <- as.numeric(item$xx)
    values <- c(history, future)
    sid <- if (!is.null(item$st)) as.character(item$st) else if (!is.null(item$sn)) as.character(item$sn) else sprintf("%s_%04d", benchmark, i)
    period <- if (!is.null(item$period)) toupper(as.character(item$period)) else "UNKNOWN"
    h <- if (!is.null(item$h)) as.integer(item$h) else length(future)
    if (h != length(future)) stop(sprintf("%s %s horizon mismatch", benchmark, sid))
    value_rows[[i]] <- data.frame(
      benchmark = benchmark,
      series_id = sid,
      position = seq_along(values) - 1L,
      value = values,
      stringsAsFactors = FALSE
    )
    meta_rows[[i]] <- data.frame(
      benchmark = benchmark,
      series_id = sid,
      period = period,
      n_history = length(history),
      horizon = h,
      total_length = length(values),
      stringsAsFactors = FALSE
    )
  }
  list(values = do.call(rbind, value_rows), metadata = do.call(rbind, meta_rows))
}

load(m1_path)
if (!exists("M1")) stop("M1 object not found")
load(tourism_path)
if (!exists("tourism")) stop("tourism object not found")

m1 <- extract_comp(M1, "M1")
tour <- extract_comp(tourism, "Tourism")
values <- rbind(m1$values, tour$values)
metadata <- rbind(m1$metadata, tour$metadata)
if (length(M1) != 1001) stop(sprintf("expected 1001 M1 series, got %d", length(M1)))
if (length(tourism) != 1311) stop(sprintf("expected 1311 Tourism series, got %d", length(tourism)))
if (any(!is.finite(values$value))) stop("nonfinite value found")
if (any(metadata$horizon <= 0)) stop("nonpositive horizon found")
write.csv(values, gzfile(file.path(out_dir, "m1_tourism_values.csv.gz"), "wt"), row.names = FALSE, quote = FALSE)
write.csv(metadata, file.path(out_dir, "m1_tourism_metadata.csv"), row.names = FALSE, quote = FALSE)
write.csv(as.data.frame(table(metadata$benchmark, metadata$period)), file.path(out_dir, "source_group_counts.csv"), row.names = FALSE, quote = FALSE)
