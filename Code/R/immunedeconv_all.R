
#install.packages("remotes")
#remotes::install_github("icbi-lab/immunedeconv")

#devtools::install_github('icbi-lab/immunedeconv', repos='http://cran.rstudio.com')

gene_expr <- read.csv("../Data/data_sum1.csv",sep=',',header=T,row.names = 1)

head(gene_expr)

colnames(gene_expr)
rownames(gene_expr)

rownames(gene_expr) <- gene_expr$ID

gene_expr <- gene_expr[,-1]
exprs_matrix <- as.matrix(gene_expr)
head(exprs_matrix)

library(tidyr)
library(immunedeconv)

res = deconvolute(exprs_matrix, "quantiseq")

res = deconvolute(exprs_matrix, "quantiseq") %>%
  map_result_to_celltypes(c("T cell CD4+"), "quantiseq")

knitr::kable(res, digits=2)

res1 <- deconvolute(your_mixture_matrix, "mcp_counter")

res2 <- deconvolute(your_mixture_matrix, "xcell")

res3 <- deconvolute(your_mixture_matrix, "epic")

## method: 
#quantiseq
#timer
#cibersort
#cibersort_abs
#mcp_counter
#xcell
#epic

