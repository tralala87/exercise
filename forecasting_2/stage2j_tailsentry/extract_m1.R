args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) stop("usage: extract_m1.R M1.rda values.csv.gz metadata.csv")
load(args[[1]])
if (!exists("M1")) stop("M1 object not found")
if (length(M1) != 1001) stop(sprintf("expected 1001 series, got %d", length(M1)))
value_rows <- vector("list", length(M1))
meta_rows <- vector("list", length(M1))
for (i in seq_along(M1)) {
  item <- M1[[i]]
  values <- c(as.numeric(item$x), as.numeric(item$xx))
  value_rows[[i]] <- data.frame(
    series_id = as.character(item$st),
    position = seq_along(values) - 1L,
    value = values,
    stringsAsFactors = FALSE
  )
  meta_rows[[i]] <- data.frame(
    series_id = as.character(item$st),
    series_name = as.character(item$sn),
    period = as.character(item$period),
    type = as.character(item$type),
    n_history = length(item$x),
    horizon = length(item$xx),
    total_length = length(values),
    description = as.character(item$description),
    stringsAsFactors = FALSE
  )
}
values <- do.call(rbind, value_rows)
metadata <- do.call(rbind, meta_rows)
connection <- gzfile(args[[2]], "wt")
write.csv(values, connection, row.names = FALSE, quote = FALSE)
close(connection)
write.csv(metadata, args[[3]], row.names = FALSE, quote = TRUE)
