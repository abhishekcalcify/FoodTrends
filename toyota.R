# set working directory 
setwd("/Users/abhi/Documents/RAssignments/PA1") 

# Package for handling and visualizing missing data (missmap) 
library(lattice)
library(ggplot2)
library(Amelia)
# corrplot: The corrplot package is a graphical display of a correlation
library(corrplot)
# Package for drawing illustrations of correlation plots 
library(psych)
# caret: Package contians functions to streamline the model training process
library(caret)

library(plotly)

library(dplyr)

# rpart: The corrplot package is a Recursive partitioning for classification and regression trees
library(rpart)
# rpart.plot: Package for plotting decision trees
library(rpart.plot)

# load data set
corolla = read.csv(file="ToyotaCorolla.csv", header=TRUE, sep=",")

corolla_new = read.csv(file="ToyotaCorolla.csv", header=TRUE, sep=",")

head(corolla)

summary(corolla$Price)

# understand the distribution of prices
corolla_price <- corolla$Price
par(mfrow=c(1,2))
hist(corolla_price, col="blue", main="Histogram", ylim=c(0, 600), breaks=11) 
plot(density(corolla_price, na.rm = TRUE), col="blue", main="Density")

# check for missing values
missmap(corolla, col=c('yellow','blue'),y.at=1,y.labels='',legend=TRUE)

# transform Fuel type to numerical
corolla_new$Fuel_Type <- factor(corolla_new$Fuel_Type, levels=c("Diesel","Petrol", "CNG"), labels=c(1,2,3))
corolla_new$Fuel_Type <- as.numeric(as.character(corolla_new$Fuel_Type)) 

str(corolla_new)

View(corolla_new)
summary(corolla_new)

# correlation plot
corrplot(cor(corolla_new))

# Dimension reduction 

# remove model names as its not significant for prediction
reject_model <- names(corolla_new) %in% c("Model") 
corolla_new <- corolla_new[!reject_model]

# Price vs Cylinders
plot(density(corolla_new$Price), col="red", xlim=c(0, 80), ylim=c(0, 0.5), main="Price vs Cyliders")
lines(density(corolla_new$Cylinders, na.rm = TRUE), col="green")

# As all the cars have same cylinders
reject_cylinders <- names(corolla_new) %in% c("Cylinders") 
corolla_new <- corolla_new[!reject_cylinders]

View(corolla_new)

# correlation matrix
cor(corolla_new, method = "spearman", use = "pairwise.complete.obs") 

# correlation plot
corrplot(cor(corolla_new))

# create correlation matrix
M <- data.matrix(corolla_new)
corrM <- cor(M)
corrM

# create highly correlated matrix
highlyCorrM <- findCorrelation(corrM, cutoff=0.5)
names(corolla_new)[highlyCorrM]

corolla_new <- data.frame(M[,-highlyCorrM])
corolla_new$Price <- corolla$Price

# correlatoin plots
corrplot(cor(corolla_new))

pairs.panels(corolla_new, col="red")

# Price vs Cylinders
plot(density(corolla$Price), col="red", main="Price vs Age")
lines(density(corolla$Age_08_04, na.rm = TRUE), col="green")

plot(density(corolla$Price),  ylim=c(0, 1700), col="red", main="Price vs  HP")
lines(density(corolla$HP, na.rm = TRUE), col="green")

# Price vs Cylinders
plot(density(corolla$Price),  ylim=c(0, 1700), col="red", main="Price vs  Id")
lines(density(corolla$Id, na.rm = TRUE), col="green")

# data partitioning 

smp_size <- floor(2/3 * nrow(corolla_new)) 
set.seed(2)

corolla_new <- corolla_new[sample(nrow(corolla_new)), ]
corrola_train <- corolla_new[1:smp_size, ]
corolla_test <- corolla_new[(smp_size+1):nrow(corolla_new), ]
str(corolla_test)

# Regression modelling

corrplot(cor(corrola_train))

formula = Price ~.

rmodel <- lm(formula = formula, data = corrola_train)
summary(rmodel)
# Residual standard error: 1719 on 933 degrees of freedom
# Multiple R-squared:  0.7787,	Adjusted R-squared:  0.7732 
# F-statistic: 142.7 on 23 and 933 DF,  p-value: < 2.2e-16

View(corrola_train)

rmodel1 <- lm(Price ~ KM + Fuel_Type + Automatic + cc, data=corrola_train)
summary(rmodel1)
#Residual standard error: 2696 on 952 degrees of freedom
#Multiple R-squared:  0.4444,	Adjusted R-squared:  0.4421 
#F-statistic: 190.4 on 4 and 952 DF,  p-value: < 2.2e-16

rmodel2 <- lm(Price ~ KM + Fuel_Type + Automatic + cc + Doors + Mfr_Guarantee 
              + Guarantee_Period + Boardcomputer + CD_Player , data=corrola_train)
summary(rmodel2)
#Residual standard error: 2233 on 947 degrees of freedom
#Multiple R-squared:  0.6209,	Adjusted R-squared:  0.6173 
#F-statistic: 172.3 on 9 and 947 DF,  p-value: < 2.2e-16

rmodel3 <- lm(Price ~ KM + Fuel_Type + Automatic + Automatic_airco + Central_Lock + Mfr_Guarantee 
               + Boardcomputer + CD_Player , data=corrola_train)
summary(rmodel3)
#Residual standard error: 1926 on 948 degrees of freedom
#Multiple R-squared:  0.7175,	Adjusted R-squared:  0.7151 
#F-statistic:   301 on 8 and 948 DF,  p-value: < 2.2e-16

# predicting the price
corolla_test$predicted.Price1 <- predict(rmodel3, corolla_test)
corolla_test$predicted.Price1

# RMSE value
error <- corolla_test$Price-corolla_test$predicted.Price1
rmselm1 <- sqrt(mean(error^2))
print(paste("Root Mean Square Error: ", rmselm1))

plot(rmodel3)


# Decision Tree
dtree <- rpart(formula, data=corrola_train, method="anova")
rpart.plot(dtree, type = 4, fallen.leaves = FALSE)

print(dtree)

plotcp(dtree)

# variable importance
dtree$variable.importance

corolla_test$predicted.Price <- predict(dtree, corolla_test)
corolla_test$predicted.Price

# calculate RMSE value
error <- corolla_test$Price-corolla_test$predicted.Price
rmse <- sqrt(mean(error^2))
print(paste("Root Mean Square Error: ", rmse))

printcp(dtree)

pruned_dtree <- prune(dtree, cp = 0.021018)
rpart.plot(dtree, type = 4, fallen.leaves = FALSE)

pruned_dtree$variable.importance

corolla_test$predicted_pruned.Price <- predict(pruned_dtree, corolla_test)
corolla_test$predicted_pruned.Price

# calculate RMSE value
error <- corolla_test$Price-corolla_test$predicted_pruned.Price
rmse <- sqrt(mean(error^2))
print(paste("Root Mean Square Error: ", rmse))

