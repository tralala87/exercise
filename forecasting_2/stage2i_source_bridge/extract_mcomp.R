args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: extract_mcomp.R M3.rda values.csv.gz metadata.csv")
load(args[[1]])
if (!exists("M3")) stop("M3 object not found")
if (length(M3) != 3003) stop(sprintf("expected 3003 series, got %d", length(M3)))
value_rows <- vector("list", length(M3))
meta_rows <- vector("list", length(M3))
for (i in seq_along(M3)) {
  item <- M3[[i]]
  values <- c(as.numeric(item$x), as.numeric(item$xx))
  value_rows[[i]] <- data.frame(
    series_id = as.character(item$st),
    position = seq_along(values) - 1L,
    value = values,
    stringsAsFactors = FALSE
  )
  meta_rows[[i]] <- data.frame(
    series_id = as.character(item$st),
    period = as.character(item$period),
    n_history = length(item$x),
    horizon = length(item$xx),
    total_length = length(values),
    stringsAsFactors = FALSE
  )
}
values <- do.call(rbind, value_rows)
metadata <- do.call(rbind, meta_rows)
values_connection <- gzfile(args[[2]], "wt")
write.csv(values, values_connection, row.names = FALSE, quote = FALSE)
close(values_connection)
write.csv(metadata, args[[3]], row.names = FALSE, quote = FALSE)
