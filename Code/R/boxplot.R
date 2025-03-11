library(tidyverse)
library(rstatix)
library(ggpubr)
library(ggtext)

df<- read.csv('../../Data/scATOMIC_Macro/bayes8_fraction_c_v5.csv')


df_p_val1 <- df %>% 
  group_by(cell_type) %>% 
  t_test(formula = values ~ group) %>% 
  add_significance(p.col = 'p',cutpoints = c(0,0.001,0.01,0.05,1),symbols = c('***','**','*','ns')) %>% 
  add_xy_position(x='cell_type')

windowsFonts(Arial = windowsFont("Arial"))

  

ggplot()+
  geom_boxplot(data = df,mapping = aes(x=cell_type,y=values,fill=group),width=0.5)+
  scale_fill_manual(values = c('#DB6A68','#8FB4DC'))+
  stat_pvalue_manual(df_p_val1,label = '{p.signif}',tip.length = 0)+
  labs(x='',y='Cell Type Fraction')+
  guides(fill=guide_legend(title = 'Group'))+
  theme_test()+
  theme(axis.text = element_text(color = 'black'),
        plot.caption = element_markdown(face = 'bold'),
        legend.position = 'top',
        legend.direction = 'horizontal')+theme(text = element_text(family = "Arial"))
  
  theme(axis.text.x = element_text(angle = 60, hjust = 1))
  scale_y_continuous(limits = c(0,1),breaks = seq(0,40,10),expand = c(0,0))