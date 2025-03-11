
library(ggplot2)
library(limma)
library(pheatmap)
library(ggsci)
lapply(c('clusterProfiler','enrichplot','patchwork'), function(x) {library(x, character.only = T)})
library(org.Hs.eg.db)
library(patchwork)
library(ggVolcano)
library(ggsci)

#load data
# group
data_label <- read.csv("../../Data/scATOMIC_Macro/group1.csv",header=T,row.names = 1)
# before deconvolution
#data_expr <- read.csv("../../Data/scATOMIC_Macro/breast_expr_c88_new.csv",header=T)

# after deconvolution
data_expr <- read.csv("../../Data/scATOMIC_Macro/breast_expr_bayes8.csv",header=T)


print(head(data_expr))

#data_expr=read.table(expFile,sep="\t",header=T,check.names=F)
rt=as.matrix(data_expr)
rownames(rt)=rt[,1]
exp=rt[,2:ncol(rt)]
dimnames=list(rownames(exp),colnames(exp))
data=matrix(as.numeric(as.matrix(exp)),nrow=nrow(exp),dimnames=dimnames)
data=avereps(data)

# group by labels

group_data = t(data_label)

group <- factor(group_data,levels = c("High","Low"))



# DEG analysis
design <- model.matrix(~0+group)
colnames(design) <- levels(group)

fit <- lmFit(data,design)

cont.matrix<-makeContrasts(High-Low,levels=design)
fit2 <- contrasts.fit(fit, cont.matrix)
fit2 <- eBayes(fit2)
deg=topTable(fit2,adjust='fdr',number=nrow(data))
Diff=deg


#save results
DIFFOUT=rbind(id=colnames(Diff),Diff)
write.table(DIFFOUT,file=paste0("../../Results/GSEA/1.","DIFF_all_after.xlsx"),sep="\t",quote=F,col.names=F)


# DEG top 30
Diff=Diff[order(as.numeric(as.vector(Diff$logFC))),]
diffGene=as.vector(rownames(Diff))
diffLength=length(diffGene)
afGene=c()
if(diffLength>(60)){
  afGene=diffGene[c(1:30,(diffLength-30+1):diffLength)]
}else{
  afGene=diffGene
}
afExp=data[afGene,]

Type1=as.data.frame(group)

rownames(Type1) = colnames(data_expr)[2:length(colnames(data_expr))]

Type1=Type1[order(Type1$group,decreasing = T),,drop=F]
Type=Type1[,1]

names(Type)=rownames(Type1)

Type=as.data.frame(Type)

anncolor=list(Type=c(High="red",Low="blue"  ))

pdf(file=paste0("D:/1.June/singlecell/Bayes_CNN/Results/new/GSEA/1.", "DIFF_heatmap1.pdf"),height=7,width=6)
pheatmap(afExp[,rownames(Type1)],                                                                      #热图数据
         annotation=Type,                                                            #分组
         color = colorRampPalette(c("blue","white","red"))(50),     #热图颜色
         cluster_cols =F,                                                           #不添加列聚类树
         show_colnames = F,                                                         #展示列名
         scale="row", 
         fontsize = 10,
         fontsize_row=6,
         fontsize_col=8,
         annotation_colors=anncolor
)
dev.off()


adjP=0.05
aflogFC=0.5
Significant=ifelse((Diff$P.Value<adjP & abs(Diff$logFC)>aflogFC), ifelse(Diff$logFC>aflogFC,"Up","Down"), "Not")
# draw 
p = ggplot(Diff, aes(logFC, -log10(P.Value)))+
  geom_point(aes(col=Significant),size=4)+               
  scale_color_manual(values=c(pal_npg()(2)[2], "#838B8B", pal_npg()(1)))+
  labs(title = " ")+
  theme(plot.title = element_text(size = 16, hjust = 0.5, face = "bold"))+
  geom_hline(aes(yintercept=-log10(adjP)), colour="gray", linetype="twodash",size=1)+
  geom_vline(aes(xintercept=aflogFC), colour="gray", linetype="twodash",size=1)+
  geom_vline(aes(xintercept=-aflogFC), colour="gray", linetype="twodash",size=1)

p
# draw (another type)
point.Pvalue=0.01
point.logFc=1.5

Diff$symbol=rownames(Diff)
pdf(paste0("../../Results/GSEA/1.", "DIFF_vol2.pdf"),width=7,height=6)
p=p+theme_bw()
for_label <- Diff %>% 
  filter(abs(logFC) >point.logFc & P.Value< point.Pvalue )
p+geom_point(size = 1.5, shape = 1, data = for_label) +
  ggrepel::geom_label_repel(
    aes(label = symbol),
    data = for_label,
    color="black",
    label.size =0.1
  )
dev.off()

######## Draw (new type) #############

deg_data1 = Diff
logFC <- Diff$logFC
log2FC <- logFC/log(2)
deg_data1$log2FC <- log2FC

# use the function -- add_regulate to add a regulate column 
# to the DEG result data. 
data <- add_regulate(deg_data1, log2FC_name = "log2FC",
                     fdr_name = "adj.P.Val",log2FC = 1, fdr = 0.05)

# plot
ggvolcano(data, x = "log2FoldChange", y = "padj",
          label = "symbol", label_number = 10, output = FALSE, legend_position="UR")#+xlim(c(-0.00001,0.00001))


####### cont ############

logFC_t=0
deg$g=ifelse(deg$P.Value>0.05,'stable',
             ifelse( deg$logFC > logFC_t,'UP',
                     ifelse( deg$logFC < -logFC_t,'DOWN','stable') )
)
table(deg$g)

deg$symbol=rownames(deg)
df <- bitr(unique(deg$symbol), fromType = "SYMBOL",
           toType = c( "ENTREZID"),
           OrgDb = org.Hs.eg.db)
DEG=deg
DEG=merge(DEG,df,by.y='SYMBOL',by.x='symbol')
data_all_sort <- DEG %>% 
  arrange(desc(logFC))

geneList = data_all_sort$logFC 
names(geneList) <- data_all_sort$ENTREZID 
head(geneList)

# GSEA-KEGG
kk2 <- gseKEGG(geneList     = geneList,
               organism     = 'hsa',
               nPerm        = 10000,
               minGSSize    = 10,
               maxGSSize    = 200,
               pvalueCutoff = 0.05,
               pAdjustMethod = "none" )
class(kk2)
colnames(kk2@result)
kegg_result <- as.data.frame(kk2)
rownames(kk2@result)[head(order(kk2@result$enrichmentScore))]
af=as.data.frame(kk2@result)
write.table(af,file=paste0("../../Results/GSEA/2.","all_GSEA_after_new11.xls"),sep="\t",quote=F,col.names=T)

# select to show
pdf(paste0("D:/1.June/singlecell/Bayes_CNN/Results/new/GSEA/2.","gsea_5_after_new11.pdf"),width = 10,height = 10)
myenrichplot::gseaplot2(kk2,#geneSetID = rownames(kk2@result)[c(head(order(kk2@result$enrichmentScore),num),tail(order(kk2@result$enrichmentScore),num))],
                        #c('hsa04152','hsa04657','hsa04210','hsa04612','hsa04217'), 
                        #c('hsa04152','hsa04512','hsa04974','hsa04612','hsa04510'),
                        c('hsa04152','hsa04210','hsa04657','hsa04612','hsa04217','hsa04611'),
                        pvalue_table = T, #显示p值
                        color = ggsci::pal_npg("nrc", alpha = 0.9)(6), # 修改配色，这里调用的是ggsci包里的lancet配色，后面的数字代表多少个颜色
                        title = 'KEGG-GSEA')

dev.off()
dev.new()


####### barplot #########
library(ggplot2)
library(tidyverse)

A <- read.csv('../../Results/new/GSEA/after_select20.csv', header = T)

A$group <- ''
A$group[which(A$NES >0)]='up'
A$group[which(A$NES <0)]='down'

ggplot(A,aes(reorder(Description, NES),NES,fill=group))+
  geom_col()+
  theme_bw()+
  theme(panel.grid.major=element_blank(),
        panel.grid.minor=element_blank(),
        panel.border = element_blank(),
        legend.title = element_blank(),
        axis.text = element_text(color="black",size=6, family = 'sans'),
        axis.line.x = element_line(color='black'),
        axis.ticks.y = element_blank(),
        axis.text.y = element_blank(),
        legend.position = 'none')+
  coord_flip()+
  geom_segment(aes(y=0, yend=0,x=0,xend=20))+
  geom_text(data = A[which(A$NES>0),],aes(x=Description, y=-0.01, label=Description),
            hjust=1, size=4)+
  geom_text(data = A[which(A$NES<0),],aes(x=Description, y=0.01, label=Description),
            hjust=0, size=4)+
  #geom_text(data = A[which(A$NES>0),],aes(label=p.adjust),
  #          hjust=-0.1, size=4, color='red')+
  #geom_text(data = A[which(A$NES<0),],aes(label=p.adjust),
  #          hjust=1.1, size=4, color="red")+
  scale_fill_manual(values = c("#F39B7FFF",
                               "#4DBBD5FF"))+
  scale_x_discrete(expand = expansion(mult = c(0,0)))+
  #ylim(-0.5, 0.5)+
  labs(x='', y='Normalized Enrichment Score')
