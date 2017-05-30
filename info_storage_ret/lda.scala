import scala.collection.mutable
import org.apache.spark.mllib.clustering.{DistributedLDAModel,LDA}
import org.apache.spark.mllib.linalg.{Vector, Vectors}
import org.apache.spark.rdd.RDD
import scalax.io.{FileOps}
import java.io._

def printToFile(f: java.io.File)(op: java.io.PrintWriter => Unit) {
  val p = new java.io.PrintWriter(f)
  try { op(p) } finally { p.close() }
}
//documents from text files, 1 document per file
val corpus: RDD[String] = sc.textFile("hdfs://nameservice1/user/cs5604s16_to/wdbj_tweets/cleaned_tweets")
corpus.count()
val tweetid = corpus.map(_.toLowerCase.split("\\s")).map(y => if(y.size > 0) y(0) else 0)
val collectionid = tweetid.first.toString.substring(0,tweetid.first.toString.indexOf('-'))
println("CollectionID:::" + collectionid)
val idmap = tweetid.zipWithIndex
val idkey = idmap.map{case(k,v) => (v,k)}
val stopWords = Array("about","means","after")
val tokenized: RDD[Seq[String]] =
  corpus.map(_.toLowerCase.split("\\s")).map(_.filter( s => !stopWords.contains(s))).map(_.filter(_.length > 4).filter(_.forall(java.lang.Character.isLetter)))

// Choose the vocabulary.
//   termCounts: Sorted list of (term, termCount) pairs

val termCounts: Array[(String, Long)] =
  tokenized.flatMap(_.map(_ -> 1L)).reduceByKey(_ + _).collect().sortBy(-_._2)

//   vocabArray: Chosen vocab (removing common terms)
val numStopwords = 0
val vocabArray: Array[String] =
  termCounts.takeRight(termCounts.size - numStopwords).map(_._1)
//   vocab: Map term -> term index
val vocab: Map[String, Int] = vocabArray.zipWithIndex.toMap

// Convert documents into term count vectors
val documents: RDD[(Long, Vector)] =
  tokenized.zipWithIndex.map { case (tokens, id) =>
    val counts = new mutable.HashMap[Int, Double]()
    tokens.foreach { term =>
      if (vocab.contains(term)) {
        val idx = vocab(term)
        counts(idx) = counts.getOrElse(idx, 0.0) + 1.0
      }
    }
   (id, Vectors.sparse(vocab.size, counts.toSeq))
  }

// Set LDA parameters
val numTopics = 5
val mlda = new LDA().setOptimizer("em").setK(numTopics).setMaxIterations(100)
val ldaModel = mlda.run(documents)
// Print topics, showing top-weighted 10 terms for each topic.
val maxwords = 10
val topicIndices = ldaModel.describeTopics(maxTermsPerTopic = maxwords)
var words = ""
var weights = ""
var topicwords: Array[String] = new Array[String](numTopics)
var i = 0;
topicIndices.foreach { case (terms, termWeights) =>
  //println("TOPIC:")
    words = ""
    weights = ""
    terms.zip(termWeights).foreach { case (term, weight) =>
    words += vocabArray(term.toInt)+","
    weights += weight.toString+","
  //  println(s"${vocabArray(term.toInt)}\t$weight")
  }
  topicwords(i) = words.substring(0,words.length-1) + "\t" + weights.substring(0,weights.length-1) 
  i = i + 1
  //println()
}

topicwords.foreach(println)
var topicLabels: Set[String] = Set()
var labellist = ""
topicIndices.foreach{ case(terms, termWeights) => 
	var i = 0
	while(i < maxwords && topicLabels.contains(vocabArray(terms(i)))){
		i = i+1;
	}
	labellist += vocabArray(terms(i)) + ","
	topicLabels += vocabArray(terms(i))
}
//topicLabels.foreach(println)
//println(labellist)
labellist = labellist.substring(0,labellist.length-1)
//topicIndices.saveAsTextFile("hdfs://nameservice1/user/cs5604s16_to/shooting_small_topic_words")

var labels: Array[String] = labellist.split(",")
labels.foreach(println)

var finalTopicWords: Array[String] = new Array[String](numTopics)
var i = 0
for(i <- 0 to numTopics - 1){
	finalTopicWords(i) = labels(i)+ "\t"+ collectionid + "\t"+ topicwords(i)
}

finalTopicWords.foreach(println)
finalTopicWords.saveAsTextFile("hdfs://nameservice1/user/cs5604s16_to/shooting_small_topic_words")
val topicWordResults: RDD[String] = finalTopicWords.map(_)
topicWordResults.first
topicWordResults.count()

printToFile(new File("topicwords.txt")) { p =>
  finalTopicWords.foreach(p.println)
}



var distLDAModel = ldaModel.asInstanceOf[DistributedLDAModel]
//distLDAModel.topicDistributions.saveAsTextFile(s"doc_topic")

println("LOG_LIKELIHOOD")
distLDAModel.logLikelihood
var localModel = distLDAModel.toLocal
localModel.logPerplexity(documents)

var topics = distLDAModel.topTopicsPerDocument(k = numTopics)
var docspertopic = distLDAModel.topDocumentsPerTopic(maxDocumentsPerTopic = 10)
docspertopic.take(2).foreach(println)

var doctopic  = sc.textFile("hdfs://nameservice1/user/cs5604s16_to/doc_topic")
var result: RDD[(Long,String)] = doctopic.map(m =>(m.substring(1,m.indexOf(",")).toLong , m.substring(m.indexOf(",")+1 , m.length-1))) 
var conv = idkey.join(result).map(_._2)
var topiclist = ""
var x = 1
for(x <- 1 to numTopics){
	topiclist += "Topic" + x + ",";
}
topiclist = topiclist.substring(0,topiclist.length-1)
var temp = sc.textFile("hdfs://nameservice1/user/cs5604s16_to/doc_topic_converted")
var finalres: RDD[String] = temp.map(m => (m.substring(1,m.indexOf(",")) + "\t" + topiclist + "\t" + m.substring(m.indexOf("[")+1 , m.indexOf("]")) +"\t" + labellist));

finalres.count
finalres.first

finalres.saveAsTextFile("hdfs://nameservice1/user/cs5604s16_to/shooting_small_doc_res")

// Printing to console
println("TOPICSCOUNT" + topics.count())
val sample = topics.take(10)
sample.foreach(println)
Save and load model
ldaModel.save(sc, "one_more_model")
var sameModel = DistributedLDAModel.load(sc, "one_more_model")
var topics = sameModel.topTopicsPerDocument(k=10)
val sameModel = LDA.load(sc, "/user/cloudera/rk_model")

//Print documents, showing topic distribution.
val documentIndices = distLDAModel.topTopicsPerDocument(k = 10)
documentIndices.foreach { case (terms, termWeights) =>
  println("DOCUMENTS:")
  terms.zip(termWeights).foreach { case (term, weight) =>
    println(s"${vocabArray(term.toInt)}\t$weight")
  }
  println()
}

