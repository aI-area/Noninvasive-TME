
install.packages("pheatmap")

library(pheatmap)

df <- read.csv("../../Data/scATOMIC_Macro/fraction_heatmap.csv", row.names = 1)

select_cols <- df[, 1:8]
select_cols

annotation_col = data.frame(Subtype=df$subtype_1)
row.names(annotation_col) <- rownames(select_cols)

subtypecolor <- c(nonTNBC = "#f38181", TNBC = "#aa96da")

pheatmap(as.matrix(select_cols),
         name = "values",
         cluster_rows = F,cluster_cols = F,
         color=colorRampPalette(c("navy","white","firebrick3"))(100),
         show_colnames = T,border_color = NA,
         scale = "row",
         #legend = FALSE,
         #legend_labels = NA,
         #annotation_legend = F,
         show_rownames =F,
         #annotation_col = annotation_col, 
         annotation_row = annotation_col,
         annotation_colors = list(Subtype = subtypecolor))

ggsave('../../Results/plot/heatmap.png',width = 5, height = 5)




###### PCA Biplot #######

library(devtools)
#install_github("vqv/ggbiplot")
library(ggbiplot)


df <- read.csv("../../Data/scATOMIC_Macro/fraction_heatmap.csv", row.names = 1)

select_cols <- df[, 1:8]
select_cols

# PCA
data_all.pca <- prcomp(select_cols, scale. = TRUE)

tiff("../../Results/plot/pca_biplot.tif",width = 2000,height = 1500)

ggbiplot(data_all.pca, # PCA Results
         choices = c(1,2), 
         obs.scale = 1, 
         var.scale = 1, 
         var.axes = TRUE, 
         groups = df$subtype_1, 
         ellipse = TRUE, 
         ellipse.prob = 0.95, 
         circle = F) +  
  theme_bw() +
  #geom_line(linewidth=1)+
  #theme(panel.border =1) +
  theme(legend.direction = 'horizontal', 
        legend.position = 'top',
        #legend.text = element_text(size = 14),
        #legend.title = element_text(size = 12)
        )
dev.off()



