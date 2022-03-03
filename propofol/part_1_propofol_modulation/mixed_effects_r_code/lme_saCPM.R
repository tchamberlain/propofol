library(lme4)
library(lmerTest)
library(dplyr) 
library(optimx)
library(apastats)
library(ggplot2)
rm(list=ls())

# Read and reformat data
setwd('/Users/taylorchamberlain/code/propofol_paper/propofol/part_1_propofol_modulation/mixed_effects_r_code')
d <- read.table("./lme_saCPM.csv", sep=",", head=T) 


d$subject <- factor(d$subject)
d$rank <- factor(d$sedation_level)
d$state_factor <- factor(d$state, levels=c("awake","recovery", "mild","deep")) 

contrasts(d$state_factor) = contr.poly(4)
options(contrasts=c("contr.sum","contr.poly"))

# Normalize the data
d$high <- scale(d$high) 
d$low  <- scale(d$low)

# Run the mixed effects models
m.high <- lmer(high ~ state_factor * task + (1 | subject), data=d, verbose=FALSE,
               control = lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B')))

m.low <- lmer(low ~ state_factor * task + (1 | subject), data=d, verbose=FALSE,
              control = lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B')))

# ADD RANDOM SLOPES in addition to random intercepts
# Run the mixed effects models
# m.high1 <- lmer(high ~ state_factor * task + (1 + state_factor  | subject), data=d, verbose=FALSE,
#                control = lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B')))
# 
# 
# m.high1 <- lmer(high ~ state_factor * task + (1 + task  | subject), data=d, verbose=FALSE,
#                 control = lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B')))
# 
# m.low1 <- lmer(low ~ state_factor * task + (1 + task  | subject), data=d, verbose=FALSE,
#               control = lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B')))
# 
# m.low1 <- lmer(low ~ state_factor * task + (1 + state_factor * task | subject), data=d, verbose=FALSE,
#               control = lmerControl(optimizer ='optimx', optCtrl=list(method='L-BFGS-B')))
# 
# anova(m.high1)
# anova(m.low1)

summary(m.high)
anova(m.high)


summary(m.low)
anova(m.low)



