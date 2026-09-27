# The datasets, from the CSVs the Python package ships (python/functai/datasets).
#   Rscript data-raw/datasets.R   (from r/)
read <- function(name) {
  x <- utils::read.csv(file.path("..", "python", "functai", "datasets", paste0(name, ".csv")),
                       stringsAsFactors = FALSE, na.strings = "", encoding = "UTF-8")
  tibble::as_tibble(x)
}
tickets <- read("tickets")
field_notes <- read("field_notes")
field_notes$date <- as.Date(field_notes$date)
refunds <- read("refunds")
refunds$final_sale <- as.logical(refunds$final_sale)
refunds$state <- factor(refunds$state, levels = c("unopened", "opened_unused", "used", "damaged", "wrong_item", "faulty"))
refunds$decision <- factor(refunds$decision, levels = c("approve", "deny"))
save(tickets, file = "data/tickets.rda", compress = "xz", version = 2)
save(field_notes, file = "data/field_notes.rda", compress = "xz", version = 2)
save(refunds, file = "data/refunds.rda", compress = "xz", version = 2)
