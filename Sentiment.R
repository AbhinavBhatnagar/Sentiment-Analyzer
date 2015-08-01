#Sentiment Analysis Extended
#Fetch the data from MongoDB
#Check if the language is english
#Remove Punctuation, clean the data
#Create Feature vector in form of a document term matrix
#train Naive Bayes classifier and predict
working <- "~/Desktop/Rdata/"
setwd(working)

#check the WD
getwd()

#load my workspace
load("workspace.Rdata")

library(rmongodb)
library(NLP)
library(SparseM)
library(tm)
library(RCurl)
library(MASS)
library(klaR)
library(zoo);
library(xts);
library(TTR);
library(RJSONIO)
library(stringr)
library(tm)
library(RColorBrewer)
library(wordcloud)
library(rpart)
library(ggplot2); 
library(lattice);
library(caret); 
library(RTextTools)
library(e1071)
library(RWeka)
library(randomForest)
library(quantmod);
library(forecast);


mongo_data <- mongo.create(host = "localhost")
#print(mongo.get.databases(mongo_data))
#print(mongo.get.database.collections(mongo_data, 'paypal_dataset'))
namespace <- mongo.get.database.collections(mongo_data, 'paypal_dataset')
#tmp <- mongo.findOne(mongo_data, ns= namespace)
#class(tmp)
#tmp <- mongo.bson.to.list(tmp)
cursor <- mongo.find(mongo_data, ns= namespace)
tweets_data = data.frame(stringsAsFactors = F)
while (mongo.cursor.next(cursor)) {
  tmp <- mongo.bson.to.list(mongo.cursor.value(cursor))
  tmp.df <- as.data.frame(t(unlist(tmp)), stringsAsFactors = FALSE)
  tweets_data <- rbind(tweets_data, tmp.df)
}
backup <- tweets_data
tweets_data <- backup
#save(file = "~/Desktop/Rdata/backup.Rdata", backup)
#load(file = "~/Desktop/Rdata/backup.Rdata", backup)
tweets_data <- tweets_data[,-1]


class(tweets_data[,2])
head(tweets_data[,2])

clean_function <- function(sentence){
sentence = gsub("(RT|via)((?:\\b\\W*@\\w+)+)", "", sentence)
sentence = gsub("@\\w+", " ", sentence)
sentence = gsub("\n", " ", sentence)
sentence = gsub("[[:punct:]]", "", sentence)
sentence = gsub("[[:digit:]]", "", sentence)
sentence = gsub("http\\w+", " ", sentence)
#sentence = gsub("[ t]{2,}", " ", sentence)
sentence = gsub("^\\s+|\\s+$", "", sentence)
sentence = gsub("amp", "", sentence)
}

sampled <- lapply(tweets_data[,2], clean_function)
#head(tweets_data[,2])
head(sampled)
tweets_data$tweet_text <- sampled
head(tweets_data[,2])


#################
sample <- tweets_data
sample <- unique(tweets_data)
#z1 <- sample$tweet_score > 0
#sample[z1,]$tweet_polarity <- 'Positive'
#z1 <- sample$tweet_score == 0.0
#sample[z1,]$tweet_polarity <- 'Neutral'
#z1 <- sample$tweet_score < 0
#sample[z1,]$tweet_polarity <- 'Negative'
table(sample$tweet_polarity)

length(which(sample[sample$tweet_score < -0.4,]$tweet_score < -0.1))

min(sample$tweet_score)
max(sample$tweet_score)
sample$tweet_polarity <- as.factor(sample$tweet_polarity)
levels(sample$tweet_polarity)

corpus <- Corpus(VectorSource(tolower((c(tweets_data$tweet_text)))))
BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 2, max = 2))
doc <- DocumentTermMatrix(corpus, control=list(tokenize=BigramTokenizer))

matrix = create_matrix(tolower(sample[, 2]), language = "english", removeStopwords = TRUE, 
                       removeNumbers = TRUE, stemWords = FALSE, tm::weightTfIdf)
mat = as.matrix(matrix)


matter <- mat[, colSums(mat) > 18]
mat <- matter


#mat$tweet_polar <- sample$tweet_polarity
inTrain <- createDataPartition(y=sample$tweet_polarity, p=0.7, list = F)
training <- mat[inTrain,]
process_obj  <- preProcess(training, method = c("center", "scale"))
testing <- mat[-inTrain,]
training_data <- predict(process_obj, training)
testing_data <- predict(process_obj, testing)
table(sample[inTrain,]$tweet_polarity)
table(sample[-inTrain,]$tweet_polarity)
dim(training_data)
dim(testing_data)

classifier <- naiveBayes(training_data, sample[inTrain,]$tweet_polarity, trainControl(method="repeatedcv", number=10, repeats = 3))

predicted <- predict(classifier, testing_data)

confusionMatrix(predicted, sample[-inTrain,]$tweet_polarity)

#
#
#train_control <- trainControl(method="LOOCV", number=10, repeats=3)
#classifier <- svm(training, sample[inTrain,]$tweet_polarity)
#classifier <- randomForest(training, sample[inTrain,]$tweet_polarity)
#table( predicted, sample[-inTrain,]$tweet_polarity)