library(lme4)
library(lmerTest)
library(dplyr) 
library(optimx)
library(apastats)
library(ggplot2)
rm(list=ls())

# Read and reformat data

# Read and reformat data
setwd('/Users/taylorchamberlain/code/propofol_paper/propofol/part_1_propofol_modulation/mixed_effects_r_code')
d <- read.table("./lme_greene_gF.csv", sep=",", head=T)


d$subject <- factor(d$subject)
d$rank <- factor(d$sedation_level)

d$state_factor <- factor(d$state, levels=c("awake","recovery", "mild","deep")) 
d$task_factor <- factor(d$task, levels=c("rest","movie")) 

contrasts(d$state_factor) = contr.poly(4)
options(contrasts=c("contr.sum","contr.poly"))

# Normalize the data
d$high <- scale(d$pos_network) 
d$low  <- scale(d$neg_network)



# Run the mixed effects models
m.high <- lmer(high ~ state_factor * task + (1 | subject), data=d, verbose=FALSE,
               control = lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B')))

m.low <- lmer(low ~ state_factor * task + (1 | subject), data=d, verbose=FALSE,
              control = lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B')))




summary(m.high)
anova(m.high)


anova(m.low)
summary(m.low)




