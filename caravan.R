rm(list = ls())
cat("\014")
gc()

# sample.kind = "Rounding" for reproducibility
set.seed(125, sample.kind = "Rounding")

# load dependencies
packs <- c(
  # plotting, grammar
  "tidyverse",
  # parallel computing
  "doMC","parallel",
  # modeling tools: roc/auc, elastic-net, RF  
   "pROC", "glmnet", "randomForest", "caret",
  # misc utilities
  "ggpubr", "readr", "scales")

# function to load packages and install as needed
loadpack <- function(pkg){
  new.pkg <- pkg[!(pkg %in% installed.packages()[, "Package"])]
   if (length(new.pkg))
     install.packages(new.pkg, dependencies = TRUE)
   sapply(pkg, require, character.only = TRUE)
}
# call the function
loadpack(packs)

# set the wd to file location
file_path <- dirname(rstudioapi::getActiveDocumentContext()$path) 
setwd(file_path)

# parallel computing: how many cores does your processor have?
num_cores = 8 
registerDoMC(cores = num_cores)
############################################################################
#                Read In & Prepare the Dataset for Analysis                #
############################################################################
program.start = proc.time()

# download the data if the csv isn't in the current working directory
if(!("caravan-insurance-challenge.csv" %in% dir())){
  dl_path <- "https://www.kaggle.com/datasets/uciml/caravan-insurance-challenge/download?datasetVersionNumber=1"
  download.file(dl_path, destfile = "data.zip")
  data <- 
    read_csv(unz("data.zip", "caravan-insurance-challenge.csv")) %>%
    as.data.frame() %>%
    filter(ORIGIN == 'train') %>%
    mutate(MOSTYPE = ifelse(MOSTYPE == 32, 5, MOSTYPE)) %>%
    select(-ORIGIN)
    
} else{
    data <- 
      read_csv("caravan-insurance-challenge.csv") %>% 
      as.data.frame() %>%
      filter(ORIGIN == 'train') %>%
      mutate(MOSTYPE = ifelse(MOSTYPE == 32, 5, MOSTYPE)) %>%
      select(-ORIGIN)
  }



# split into train & test at .90/.10 split
idx = caret::createDataPartition(data$CARAVAN, p = 0.1, list = F)
train = data[-idx,]
test  = data[idx, ]

rm(idx)

# All vars categorical. Use model.matrix to expand and one-hot-encode the feature space

y.train <- factor(train$CARAVAN)
x.train <- train[,-86]
x.train <- lapply(x.train, 
                  \(col){
                    if(length(unique(col)) == 1 ){
                      factor(col,levels = c(min(col), min(col)+1))
                    } else{
                      factor(col)
                    }
                    
                      })
x.train <- model.matrix(~.-1, data = x.train) # ~.-1 = no intercept

y.test <- factor(test$CARAVAN)
x.test <- test[,-86]
x.test <- lapply(x.test,
                 \(col){
                    if(length(unique(col)) == 1 ){
                      factor(col,levels = c(min(col), min(col)+1))
                    } else{
                      factor(col)
                    }
                    
                      })
x.test <- model.matrix(~.-1, data = x.test) # ~.-1 = no intercept

# put it all together for use with Random Forest. 
data.full <- 
  x.train %>%
  as.data.frame() %>%
  mutate(CARAVAN = y.train)

n = nrow(x.train)
p = dim(x.train)[2]

############################################################################
#                         Initialize Storage Matrices                      #
############################################################################

# we run 50 simulations and record the performance of each model 
# using AUC as our metric of choice 

runs = 50  

## AUC Data frame
# initialize a vector to be used for the columns
auc.initialize <- rep(0, runs)  

Run   = seq(1, runs)
ElNet = auc.initialize
Lasso = auc.initialize
Ridge = auc.initialize
RF    = auc.initialize

auc.df <- data.frame(Run         = Run,
                     ElNet.train = ElNet,  # 2
                     Lasso.train = Lasso,  # 3
                     Ridge.train = Ridge,  # 4
                     RF.train    = RF,     # 5
                     ElNet.test  = ElNet,  # 6
                     Lasso.test  = Lasso,  # 7
                     Ridge.test  = Ridge,  # 8
                     RF.test     = RF)     # 9

## CV Times Data frame
times = data.frame(Run = Run, 
                   ElNet = rep(0, runs), 
                   Lasso = rep(0, runs), 
                   Ridge = rep(0, runs),
                   RF    = rep(0, runs))

## Lambdas Data Frame
lambdas.df <- data.frame(Run   = seq(1,50,1), 
                         ElNet = rep(0,runs),
                         Lasso = rep(0,runs),
                         Ridge = rep(0,runs))

# Individual run times for each loop
run.times <- data.frame(Loop = seq(1,50),
                        Time = rep(0,50))


# alpha parameters
elnet.alpha = 0.5
lasso.alpha = 1
ridge.alpha = 0
############################################################################
#                             50-Run Simulation                            #
############################################################################
sim.start <- proc.time()
for(i in 1:runs){
  print(paste0("Cross Validation Number ",i))
  
  loop.start = proc.time()
 
  
  ##### Sample & Create Train/Test Indices #####
  n_obs         <- seq(1:n)
  sample.size   <- floor(.90*n)
  train.indices <- sample(seq_len(n), size=sample.size)
  test.indices  <- !(n_obs%in%train.indices)
  
  ##### Partition Training Data #####
  print("Partitioning Data")
  X.train = x.train[train.indices,]
  Y.train = y.train[train.indices]

  X.test = x.train[test.indices,]
  Y.test = y.train[test.indices]
  
  ##### Cross Validation #####
  print("Beginning Elastic Net Cross Validation")
  
  # Elastic Net
  elnet.start <- proc.time()
  elnet.cv    <- cv.glmnet(X.train, Y.train,
                           parallel     = TRUE, 
                           family       = "binomial",
                           alpha        = elnet.alpha, 
                           type.measure = "auc")
  elnet.end   <- proc.time()
  elnet.time  <- elnet.end[3]-elnet.start[3]

  times$ElNet[i]  <- elnet.time
  
  # Lasso 
  print("Beginning Lasso Cross Validation")
  lasso.start <- proc.time()
  lasso.cv    <- cv.glmnet(X.train, Y.train, 
                           parallel     = TRUE,
                           family       = "binomial",
                           alpha        = lasso.alpha, 
                           type.measure = "auc")
  lasso.end   <- proc.time()
  lasso.time  <- lasso.end[3]-lasso.start[3]
  
  times$Lasso[i]  <- lasso.time
  
  # Ridge
  print("Beginning Ridge Cross Validation Number")
  ridge.start <- proc.time()
  ridge.cv    <- cv.glmnet(X.train, Y.train, 
                           parallel     = TRUE,
                           family       = "binomial",
                           alpha        = ridge.alpha,
                           type.measure = "auc")
  ridge.end   <- proc.time()
  ridge.time  <- ridge.end[3]-ridge.start[3]
  
  times$Ridge[i]  <- ridge.time
  
  
  if(i == 5){
    writeLines("Plotting Cross Validation Curves")
    par(mfrow = c(3,1))
    plot(elnet.cv)
    title(main="Elastic-Net Cross Validation Curve",line = 3)
    plot(lasso.cv)
    title(main="Lasso Cross Validation Curve",line = 3)
    plot(ridge.cv)
    title(main="Ridge Cross Validation Curve",line = 3)
  }
  
  
  lambdas.df$ElNet[i] <- elnet.cv$lambda.min
  lambdas.df$Lasso[i] <- lasso.cv$lambda.min
  lambdas.df$Ridge[i] <- ridge.cv$lambda.min
  
  
  # Pull out the parameters
  print("Extracting the Elastic Net Beta Coefficients")
  elnet.index   <- which.max(elnet.cv$cvm)
  elnet.beta    <- as.vector(elnet.cv$glmnet.fit$beta[, elnet.index])
  elnet.beta0   <- elnet.cv$glmnet.fit$a0[elnet.index]
  
  print("Extracting the Lasso Beta Coefficients")
  lasso.index   <- which.max(lasso.cv$cvm)
  lasso.beta    <- as.vector(lasso.cv$glmnet.fit$beta[, lasso.index])
  lasso.beta0   <- lasso.cv$glmnet.fit$a0[lasso.index]
  
  print("Extracting the Ridge Beta Coefficients")
  ridge.index   <- which.max(ridge.cv$cvm)
  ridge.beta    <- as.vector(ridge.cv$glmnet.fit$beta[, ridge.index])
  ridge.beta0   <- ridge.cv$glmnet.fit$a0[ridge.index]
  
  ##### Calculate AUC #####
  print("Calculating AUC")
  print("Calculating Distances From the Hyper Plane")
  # Calculate Distances from the hyper-plane
  xtb.elnet.train     <- X.train%*%elnet.beta + elnet.beta0
  xtb.lasso.train     <- X.train%*%lasso.beta + lasso.beta0
  xtb.ridge.train     <- X.train%*%ridge.beta + ridge.beta0
  
  xtb.elnet.test      <- X.test%*%elnet.beta + elnet.beta0
  xtb.lasso.test      <- X.test%*%lasso.beta + lasso.beta0
  xtb.ridge.test      <- X.test%*%ridge.beta + ridge.beta0
  
  print("Calculating The Probability Matrices")
  # use distances to calculate probabilities
  elnet.probs.train = exp(xtb.elnet.train)/(1+exp(xtb.elnet.train))
  elnet.probs.test = exp(xtb.elnet.test)/(1+exp(xtb.elnet.test))
  
  lasso.probs.train = exp(xtb.lasso.train)/(1+exp(xtb.lasso.train))
  lasso.probs.test = exp(xtb.lasso.test)/(1+exp(xtb.lasso.test))
  
  ridge.probs.train = exp(xtb.ridge.train)/(1+exp(xtb.ridge.train))
  ridge.probs.test = exp(xtb.ridge.test)/(1+exp(xtb.ridge.test))
  
  # Calculate AUC for El-net, Lasso, Elastic Net & Store them for use later
  print("Storing AUC")
  elnet.train.auc <- auc(roc(as.factor(Y.train), c(elnet.probs.train)))
  lasso.train.auc <- auc(roc(as.factor(Y.train), c(lasso.probs.train)))
  ridge.train.auc <- auc(roc(as.factor(Y.train), c(ridge.probs.train)))
  
  elnet.test.auc <- auc(roc(as.factor(Y.test), c(elnet.probs.test)))
  lasso.test.auc <- auc(roc(as.factor(Y.test), c(lasso.probs.test)))
  ridge.test.auc <- auc(roc(as.factor(Y.test), c(ridge.probs.test)))
  
  auc.df$ElNet.train[i] <- elnet.train.auc
  auc.df$Lasso.train[i] <- lasso.train.auc
  auc.df$Ridge.train[i] <- ridge.train.auc
  
  auc.df$ElNet.test[i] <- elnet.test.auc
  auc.df$Lasso.test[i] <- lasso.test.auc
  auc.df$Ridge.test[i] <- ridge.test.auc
  ##### Random Forest #####
  print("Fitting Random Forest")
  # Random Forest
  rf.start     <- proc.time()
  rf.train     <- randomForest(CARAVAN~., data = data.full[train.indices,], mtry=sqrt(p))  # model
  rf.end       <- proc.time()
  rf.time      <- rf.end[3]-rf.start[3]
  times$RF[i]  <- rf.time
  
  print("Random Forest - Calculating Train and Test AUC")
  preds.test   <- predict(rf.train, newdata = X.test, type = "vote") # test predictions 
  
  # train auc
  rf.train.auc <- auc(roc(data.full$CARAVAN[train.indices], rf.train$votes[,2]))
  # test auc
  rf.test.auc  <- auc(roc(data.full$CARAVAN[test.indices],  preds.test[,2]))       
  
  # Store AUC in the data frame above
  auc.df$RF.train[i] <- rf.train.auc
  auc.df$RF.test[i]  <- rf.test.auc
  ##### End #####
  loop.end = proc.time()
  loop.time = loop.end[3]-loop.start[3]
  run.times[i,2] <- loop.time
  print(paste0("Loop ", i, "of 50 took ", loop.time, " seconds to complete"))
}
sim.end <- proc.time()
total.sim.time <- sim.end[3]-sim.start[3]
total.sim.time 
###############################################
###             AUC Box-Plots              ###
###############################################

# Labels
sample.train <- rep("Train", 200)
sample.test  <- rep("Test", 200)
elnet.label  <- rep("Elastic Net", 50)
lasso.label  <- rep("Lasso", 50)
ridge.label  <- rep("Ridge", 50)
rf.label     <- rep("Random Forest", 50)

# model labels
method.labels = c(elnet.label, lasso.label, ridge.label, rf.label)

# auc vals
train_auc = c(auc.df$ElNet.train, 
              auc.df$Lasso.train, 
              auc.df$Ridge.train, 
              auc.df$RF.train)
test_auc  = c(auc.df$ElNet.test, 
              auc.df$Lasso.test,
              auc.df$Ridge.test,
              auc.df$RF.test)
total_auc = c(train_auc, test_auc)

train.auc.df = data.frame(Sample = sample.train, 
                          Model = method.labels, 
                          AUC = train_auc)
test.auc.df = data.frame(Sample = sample.test, 
                         Model = method.labels, 
                         AUC = test_auc)

# Long d-frame - for ease of use with ggplot
long.df.auc = bind_rows(train.auc.df, test.auc.df)
long.df.auc %>% 
  ggplot(aes(x=Model, y = AUC, color = Model)) + 
  geom_boxplot() + 
  facet_wrap(~Sample)


auc.train.df <- data.frame(Sample = sample.train,
                           ElNet = auc.df$ElNet.train,
                           Lasso = auc.df$Lasso.train,
                           Ridge = auc.df$Ridge.train, 
                           RF    = auc.df$RF.train)
auc.test.df <- data.frame(Sample = sample.test, 
                          ElNet  = auc.df$ElNet.test,
                          Lasso  = auc.df$Lasso.test,
                          Ridge  = auc.df$Ridge.test,
                          RF     = auc.df$RF.test)

long.auc.df <- rbind(auc.train.df, auc.test.df)

##### Create The Box Plots #####

elnet.auc.box <- 
  long.auc.df %>% 
  ggplot(aes(x = Sample, y = ElNet)) +
  geom_boxplot() +
  labs(title = "Elastic Net AUC Boxplots",
       y = "AUC") +
  theme_bw()

lasso.auc.box <- 
  long.auc.df %>% 
  ggplot(aes(x = Sample, y = Lasso)) +
  geom_boxplot() +
  labs(title = "Lasso AUC Boxplots",
       y = "AUC") +
  theme_bw()

ridge.auc.box <- 
  long.auc.df %>% 
  ggplot(aes(x = Sample, y = Ridge)) +
  geom_boxplot() +
  labs(title = "Ridge AUC Boxplots",
       y = "AUC") +
  theme_bw()

rf.auc.box <- 
  long.auc.df %>% 
  ggplot(aes(x = Sample, y = RF)) +
  geom_boxplot() +
  labs(title = "Random Forest AUC Boxplots",
       y = "AUC") +
  theme_bw()


ggarrange(elnet.auc.box, lasso.auc.box,
             ridge.auc.box, rf.auc.box,
             nrow=2, ncol=2)

###############################################
### Fit the 4 models on the entire data set ###
###############################################

methods = c("Elastic Net", "Lasso", "Ridge", "Random Forest")

elnet.median.test.auc <- median(auc.df$ElNet.test)
lasso.median.test.auc <- median(auc.df$Lasso.test)
ridge.median.test.auc <- median(auc.df$Ridge.test)
rf.median.test.auc    <- median(auc.df$RF.test)

##### Storage for Median Test AUC and Full Run Times #####
median.auc <- c(elnet.median.test.auc, lasso.median.test.auc,
                ridge.median.test.auc, rf.median.test.auc)
full.times <- c(rep(0,4))


##### Elastic Net #####
elnet.start.full <- proc.time()
# cross validate lambda
elnet.full.cv    <- cv.glmnet(x.train, y.train, 
                              parallel     = TRUE,
                              family       = "binomial",
                              alpha        = elnet.alpha,
                              type.measure = "auc")
# el-net optimal lambda 
elnet.full.lambda <- elnet.full.cv$lambda.min
# elnet full model w/ optimal lambda
elnet.model <- glmnet(x.train, y.train, 
                      family="binomial",
                      alpha = elnet.alpha)
# time
elnet.end.full <- proc.time()
elnet.full.time <- elnet.end.full[3] - elnet.start.full[3]
full.times[1] <- elnet.full.time

# elnet coefficients
elnet.1se.coefs <- elnet.full.cv$glmnet.fit$beta[, which(elnet.full.cv$lambda==elnet.full.cv$lambda.1se)]
elnet.1se.coefs <- elnet.1se.coefs[elnet.1se.coefs!=0]

##### Lasso #####

lasso.start.full  <- proc.time()
# cross validate optimal lambda
lasso.full.cv     <- cv.glmnet(x.train, y.train, 
                               parallel     = TRUE,
                               family       = "binomial",
                               alpha        = lasso.alpha,
                               type.measure = "auc")
# lasso optimal lambda
lasso.full.lambda <- lasso.full.cv$lambda.min
# lasso full model w/ optimal lambda
lasso.model       <- glmnet(x.train, y.train, 
                            family  = "binomial",
                            alpha   = lasso.alpha)
# time
lasso.end.full    <- proc.time()
lasso.full.time   <- lasso.end.full[3] - lasso.start.full[3]
full.times[2]     <- lasso.full.time

# lasso coefficients - 1 standard error
lasso.1se.coefs = lasso.full.cv$glmnet.fit$beta[,which(lasso.full.cv$lambda==lasso.full.cv$lambda.1se)]
lasso.1se.coefs = lasso.1se.coefs[lasso.1se.coefs != 0 ]

##### Ridge #####

ridge.start.full  <- proc.time()
# cross validate optimal lambda
ridge.full.cv     <- cv.glmnet(x.train, y.train, 
                               parallel     = TRUE,
                               family       = "binomial",
                               alpha        = ridge.alpha,
                               type.measure = "auc")
# optimal lambda
ridge.full.lambda <- ridge.full.cv$lambda.min
# ridge full model w/ optimal lambda
ridge.model       <- glmnet(x.train, y.train,
                            family="binomial",
                            alpha = ridge.alpha)
# time
ridge.end.full    <- proc.time()
ridge.full.time   <- ridge.end.full[3] - ridge.start.full[3]
full.times[3]     <- ridge.full.time

# ridge coefficients 1-std error
ridge.1se.coefs <- ridge.full.cv$glmnet.fit$beta[,which(ridge.full.cv$lambda==ridge.full.cv$lambda.1se)]
ridge.1se.coefs <- ridge.1se.coefs[ridge.1se.coefs!=0]

###### Random Forest ######
rf.start.full     <- proc.time()
# w/ so many predictors, sqrt(p) is optimal mtry parameter
rf.full.model     <- randomForest(CARAVAN~., data = data.full, mtry = sqrt(p))
rf.end.full       <- proc.time()
rf.time           <- rf.end.full[3]-rf.start.full[3]
full.times[4]     <- rf.time

##### Collect AUC and Run Times into a Data Frame #####

auc.times.df <- data.frame(Method = methods,
                           AUC = median.auc,
                           Time = full.times)

# auc plot
auc.times.df %>%
  ggplot(aes(x=Time, y =AUC, color = Method)) + 
  geom_point()

#####  Bar plots of the standardized coefficients  #####

s = apply(x.train, 2, sd) # get the standard deviation of the variables

elnet.full.beta <- as.vector(elnet.model$beta[,which.max(elnet.full.cv$cvm)])
lasso.full.beta <- as.vector(lasso.model$beta[,which.max(lasso.full.cv$cvm)])
ridge.full.beta <- as.vector(ridge.model$beta[,which.max(ridge.full.cv$cvm)])

# multiply coefficients by sd(Variable) to standardize
elnet.coefs       <- elnet.full.beta*s  
lasso.coefs       <- lasso.full.beta*s
ridge.coefs       <- ridge.full.beta*s
# pull the importance of the variables
rf.importance     <- importance(rf.full.model)
# remove rownames from the variable importance
row.names(rf.importance) <- NULL                       


# label the variable by number for ease of display
VarNames          <- colnames(x.train)
VarNumber         <- as.character(seq(1:ncol(x.train))) 


variable.importance   <- data.frame(Variable = VarNames, 
                                    Number   = VarNumber,
                                    ElNet    = elnet.coefs, 
                                    Lasso    = lasso.coefs, 
                                    Ridge    = ridge.coefs, 
                                    RF       = rf.importance)

# Order Coefficients By desc(ElNet)
variable.importance <- 
  variable.importance[order(variable.importance$ElNet, decreasing = TRUE),]
# force ggplot to respect the order the data is sorted in. 
variable.importance$Number <- 
  factor(variable.importance$Number, 
         levels=variable.importance$Number) 


# create the plots so we can feed them to grid arrange
elnetPlot <-
  variable.importance %>% 
  ggplot(aes(x = Number, y = ElNet)) +
  geom_col() +
  labs(title = "Standardized Elastic Net Coefficients", x = "Variable", y = "Coefficient") +
  theme_bw()+
  theme(axis.text.x=element_blank())

lassoPlot = variable.importance %>% ggplot(aes(x = Number, y = Lasso)) +
  geom_col() +
  labs(title = "Standardized Lasso Coefficients", x = "Variable", y = "Coefficient") +
  theme_bw()+
  theme(axis.text.x=element_blank())

ridgePlot = variable.importance %>% ggplot(aes(x = Number, y = Ridge)) +
  geom_col() +
  labs(title = "Standardized Ridge Coefficients", x = "Variable", y = "Coefficient") +
  theme_bw() +
  theme(axis.text.x=element_blank())

rfPlot = variable.importance %>% ggplot(aes(x = Number, y = MeanDecreaseGini)) +
  geom_col() +
  labs(title = "Random Forrest Variable Importance", x = "Variable", y = "Coefficient") +
  theme_bw() +
  theme(axis.text.x=element_blank())


# arrange the plots in a single image
ggarrange(elnetPlot, lassoPlot, ridgePlot, rfPlot, nrow=4)

program.end = proc.time()
program.time = program.end[3]-program.start[3]
program.time

# save workspace for loading into the r-markdown document
file.path = paste0(dirname(file_path),"/caravan_workspace.RData")
save.image(file.path)


