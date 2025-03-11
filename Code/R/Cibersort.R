
library(CIBERSORT)
sig_matrix <- system.file("extdata", "LM22.txt", package = "CIBERSORT")

mixture_file <- "../../Data/GSE136831.txt"

results <- cibersort(sig_matrix, mixture_file)
write.csv(results, "../../Results/CIBERSORT_GSEResults.csv")


