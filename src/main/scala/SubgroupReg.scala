import org.apache.log4j.{Level, Logger}
import org.apache.spark.{SparkConf, SparkContext}
import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.util.random.BernoulliSampler

import scala.util.control.Breaks
// DenseVector默认列向量
import breeze.linalg.{DenseMatrix, cholesky, inv, max => breezemax, _}
import breeze.numerics._
import org.apache.spark.rdd.RDD
import scala.collection.mutable.ArrayBuffer
import java.io._

object SubgroupReg {
  def SoftThreashold(t:Double, lambda:Double):Double={
    if(t>abs(lambda)){
      t-abs(lambda)
    }else if(t<(-1*abs(lambda))){
      t+abs(lambda)
    }else{0.0}
  }

  def PenFunc(x:Double, penalty:String, gamma:Double, lambda:Double, a:Double, lassophi:Double=0.0):Double={
    if (penalty == "Lasso") {
      SoftThreashold(x, lambda/ a)
    } else if (penalty == "MCP") {
      if (abs(x) < (gamma * lambda)) {
        SoftThreashold(x, lambda / a) / (1 - 1 / (gamma * a))
      } else x
    } else {
      if (abs(x) < (lambda + lambda / a)) {
        SoftThreashold(x, lambda / a)
      } else if (abs(x) > (gamma * lambda)) {
        x
      } else {
        SoftThreashold(x, gamma * lambda / ((gamma - 1) * a)) / (1 - 1 / ((gamma - 1) * a))
      }
    }
  }

  def mean(a: ArrayBuffer[Double]): Double = a.sum / a.length.toDouble

  def variance(a: ArrayBuffer[Double]): Double = {
    val avg = mean(a)
    math.sqrt(a.map(x => math.pow(x - avg,2)).sum / a.length.toDouble)
  }

  def main(args: Array[String]): Unit = {
    Logger.getLogger("org").setLevel(Level.ERROR)
    val conf = new SparkConf().setAppName("DisSubgroup").setMaster("local[*]")
    val sc = new SparkContext(conf)
    // 实验设定
    //Each Group needs to have sufficient samples in at least one node
    val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(4)))
    val bernoulli = false
    val bernoulli_value = Array(-2.0, 2.0)
    val prefixfilename = "E:\\DisSubgroup\\simulation20201201\\NodeResultSubgroupSimulationAtn"
    val sim_time = 5
    val M: Int = 20
    val beta_same_num = sc.broadcast(3.0)
    val MBroad = sc.broadcast(M)
    val nm = 200
    var p = 200
    val mutp = p*(p-1)/2
    if (bernoulli){p = p + mutp}
    val N = nm*M
    val alpha = 2.0
    val a = 2.0
    val b = 2.0
    val c = 2.0
    val d = 2.0
    // 每次实验的mse和k结果
    var mse_beta_sim = new ArrayBuffer[Double]()
    var mse_mu_sim = new ArrayBuffer[Double]()
    var mse_k_sim = new ArrayBuffer[Double]()
    var mse_bic_sim = new ArrayBuffer[Double]()
    var mse_time_sim = new ArrayBuffer[Double]()
    // 更新mu时左侧逆的解
    val S = (c*(N-1)+M*a)/(M*a-c)
    val diag_inv = -1*(S+N-2)/(((N-1)-S*(N-2)-S*S)*(M*a-c))
    val undiag_inv = 1/(((N-1)-S*(N-2)-S*S)*(M*a-c))
    val Iter = 40
    val gamma = sc.broadcast(3.7)
    val lambdamin = sc.broadcast(0.019)
    val lambdastep = sc.broadcast(0.0001)
    val lambdamax = sc.broadcast(0.025)
    val lambdanum = List(lambdamin.value to lambdamax.value by lambdastep.value: _*).length
    val lassophi = sc.broadcast(0.0)
    val penalty = sc.broadcast("Lasso")
    val bicphi = sc.broadcast(10.0)
    var mseu = 0.0
    var mseb = 0.0
    var k = 0.0
    var bic = 0.0
    val tol = 0.0001
    val minportion = 0.5
    // Generate sigma and mu matrix
    var sigma = DenseMatrix.zeros[Double](rows = p, cols = p)
    for (i <- 0 until p) {
      for (j <- 0 until p) {
        sigma(i, j) = pow(0.5, abs(i - j))
      }
    }
    // 节约内存，名称不变
    sigma = cholesky(sigma)
    // 用同样的值生成beta，且仅生成一次
    val Beta = DenseVector.ones[Double](p) * beta_same_num.value
    // 生成mu，且仅生成一次
    val unif_dist_two = new Uniform(0, 1)(randBasis)
    val Mu = new DenseMatrix(rows = nm, cols = M, unif_dist_two.sample(N).map{s =>
      if (s > minportion) alpha else -1 * alpha}.toArray)
//    val Mu = new DenseMatrix(rows = nm, cols = M, unif_dist_two.sample(N).map(s =>{
//      if (s > 0.6) {alpha}
//      else if(s<0.3){-1 * alpha}
//      else{0.0}}).toArray)
    println(Mu.map(s=>s+alpha).sum / (2*alpha))
    // Delta Matrix (W) 暂时不用
    // val deltamat = sc.broadcast(DeltaMatrix(nm))
    val NodeIndex = 0 until M
    val writer = new PrintWriter(new File(prefixfilename + nm + "p" + p + "m" + M + "alpha" + alpha.toString + "phi" + lassophi.value + "pen" + penalty.value + "portion" + minportion + ".txt"))
    writer.println("lambda,BIC,MSEB,MSEU,K,unique_mu")
    //val writer1 = new PrintWriter(new File("D:\\DisSubgroup\\模拟结果\\NodeResultSubgroupSimulationAtn" + nm.value + "p" + p.value + "m" + M + "alpha" + alpha.value + "phi" + lassophi.value + penalty.value + ".txt"))
    //writer1.println("node,beta,u_distinct,lambda")
    for (sim_index <- 0 until sim_time) {
      println("start "+sim_index+" simulation."  )
      // 每个节点独立生成xm数据
      var SampledData: RDD[List[DenseMatrix[Double]]] = sc.parallelize(NodeIndex).map(m => {
        List(new DenseMatrix(1, 1, Array(m.toDouble)))
      })
      if(bernoulli){
        SampledData = sc.parallelize(NodeIndex).map(m => {
          val bernoulli_dist = new BernoulliSampler[Int](0.5)
          val error_dist = new Gaussian(0, 0.1)
          var xm: DenseMatrix[Double] = new DenseMatrix(rows = nm, cols = p-mutp,
            (0 until (nm * (p-mutp))).map(d => bernoulli_dist.sample()).toArray.map(s=>bernoulli_value(s)))
          for (i <- 0 until (p-1-mutp)){
            for (j <- (i+1) until (p-mutp)){
              xm = DenseMatrix.horzcat(xm, (xm(::,i):*xm(::,j)).toDenseMatrix.t)
            }
          }
          val ym: DenseMatrix[Double] = (xm * Beta + Mu(::, m) + DenseVector(error_dist.sample(nm).toArray)).toDenseMatrix
          val betaleft: DenseMatrix[Double] = inv(DenseMatrix.eye[Double](p) + (d/b) * (xm.t * xm))
          List(xm, ym, betaleft, new DenseMatrix(1, 1, Array(m.toDouble)))
        })
      }else{
        SampledData = sc.parallelize(NodeIndex).map(m => {
          val norm_dist = new Gaussian(0, 1)
          val error_dist = new Gaussian(0, 0.1)
          val xm: DenseMatrix[Double] = new DenseMatrix(rows = nm, cols = p, norm_dist.sample(nm * p).toArray)
          val ym: DenseMatrix[Double] = (xm * Beta + Mu(::, m) + DenseVector(error_dist.sample(nm).toArray)).toDenseMatrix
          val betaleft: DenseMatrix[Double] = inv(DenseMatrix.eye[Double](p) + (d/b) * (xm.t * xm))
          List(xm, ym, betaleft, new DenseMatrix(1, 1, Array(m.toDouble)))
        })
      }
      SampledData.persist()
      println("Finished generating data.")
      // 循环中各节点更新的全局变量
      // beta_m, mu_m, z_m
      var node_update_res1: List[DenseMatrix[Double]] = List(
        DenseMatrix.zeros[Double](rows = p, cols = M),
        DenseMatrix.zeros[Double](rows = nm, cols = M),
        DenseMatrix.zeros[Double](rows = nm, cols = M)
      )
      // 循环中各节点更新的全局变量
      // v_m, q_m, u_m
      var node_update_res2: List[DenseMatrix[Double]] = List(
        DenseMatrix.zeros[Double](rows = p, cols = M),
        DenseMatrix.zeros[Double](rows = nm, cols = M),
        DenseMatrix.zeros[Double](rows = nm, cols = M)
      )
      // 循环中主机更新的全局变量,由于类型不同，这里分开生成
      // beta, mu, eta
      var betahat: DenseVector[Double] = DenseVector.zeros[Double](p)
      var muhat: DenseMatrix[Double] = DenseMatrix.zeros[Double](rows = nm, cols = M)
      // 设计为只在最后收敛后计算一次而且只为了得到分组结果
      // var etahat: SparseVector[Double] = SparseVector.zeros[Double](nm*M*(nm*M-1)/2)
      // 更新mu中不可避免用到的r向量
      var r: DenseMatrix[Double] = DenseMatrix.zeros[Double](rows = nm, cols = M)
      // 更新group信息和center

      // 存储不同lambda的结果便于最终根据bic得到结果
      var beta_of_each_lambda:List[DenseMatrix[Double]] = List(DenseMatrix.zeros[Double](1,1))
      var u_distinct_of_each_lambda:List[Array[Double]] = List(Array(0.0))
      var bic_lambda_mse_k = DenseMatrix.zeros[Double](List(lambdamin.value to lambdamax.value by lambdastep.value: _*).length, 5)
      var bic_lambda_mat_index = 0
      val startTime: Long = System.nanoTime
      for (lambda <- List(lambdamin.value to lambdamax.value by lambdastep.value: _*)) {
        bic_lambda_mse_k(bic_lambda_mat_index, 1) = lambda
        println("lambda:" + lambda)
        val startTime1: Long = System.nanoTime
        for (itrind <- 0 until Iter){
          // 更新betam, mum, zm，为了避免reduce破坏顺序只能sort，collect之后再赋值,
          // 同时由于um的更新不需要beta和mu但又需要x和y所以也在这里更新
          var tmp = SampledData.map(s => {
            val xm = s.head
            val ym = s(1).toDenseVector
            val m = s(3)(0, 0).toInt
            val betam = s(2) * (
              (d/b)*(xm.t*ym - xm.t*node_update_res1(1)(::, m)-xm.t*node_update_res1(2)(::, m))
                + (1/b)*(xm.t*node_update_res2(2)(::, m))
                + betahat - (1/b)*node_update_res2.head(::, m)
              )
            val mum = (a/(a+b)) * (
              (d/a)*(ym - xm*betam - node_update_res1(2)(::, m))
                + (1/a)*node_update_res2(2)(::, m)
                + muhat(::, m)
                - node_update_res2(1)(::, m)
              )
            val zm = (1/(d+1)) * (d*(ym - xm*betam - mum) + node_update_res2(2)(::,m))
            val um = node_update_res2(2)(::, m) + d*(ym - xm * betam - zm - mum)
            (s(2)(0, 0).toInt, List(betam, mum, zm, um))
          }).sortByKey().collect()
          for (m <- 0 until M){
            node_update_res1.head(::,m) := tmp(m)._2.head
            node_update_res1(1)(::,m) := tmp(m)._2(1)
            node_update_res1(2)(::,m) := tmp(m)._2(2)
            node_update_res2(2)(::,m) := tmp(m)._2(3)
          }
          // 更新beta，mu，eta
          betahat = (1/(b*M)) * (b * sum(node_update_res1.head, Axis._1) + sum(node_update_res2.head, Axis._1))
          for (i <- 0 until M){
            for (j <- 0 until nm){
              r(j, i) = (muhat - muhat(j, i)).mapValues(s=>
                // mum只有涉及到的节点的那个i的mu才会更新，其他节点的那个i的mu都和mu一样，q则是其他都是0，
                // 只有涉及到的节点的那个i的mu才会有值
                PenFunc(x=s*(-1), penalty=penalty.value, gamma=gamma.value,
                  lambda=lambda, a=c, lassophi=lassophi.value)).sum +
                // a * (node_update_res1(1)(j, i) + (N-1) * muhat(j, i)) + node_update_res2(1)(j, i)
                a * node_update_res1(1)(j, i) + node_update_res2(1)(j, i)
            }
          }
          var change = 0.0
          for (i <- 0 until M){
            for (j <- 0 until nm){
              val muupdate = r.map(s=>s*undiag_inv).sum - r(j, i)*undiag_inv + r(j, i)*diag_inv
              change = change + abs(muupdate - muhat(j, i))
              muhat(j, i) = muupdate
            }
          }
          // 更新停止阈值检查
//          change = change / N
//          if (change < tol){
//            println("已经达到迭代停止条件")
//          }

          // 更新vm，qm, um已经在前面更新过了
          for (m <- 0 until M){
            node_update_res2.head(::,m) := node_update_res2.head(::,m) + b * (node_update_res1.head(::,m) - betahat)
            node_update_res2(1)(::,m) := node_update_res2(1)(::,m) + a * (node_update_res1(1)(::,m) - muhat(::,m))
          }
        }
        // 分组直到全部有组别
        var groupid = new ArrayBuffer[Double]()
        var groupcenter = new ArrayBuffer[Double]()
        groupid += 0.0
        groupcenter += muhat(0, 0)
        var grouphat = muhat
        var newcenter = true
        while (newcenter){
          // 对于没有组别的样本按照之前组别的均值和新组别的第一个值来分组
          grouphat = grouphat.mapValues(s =>{
            var groupres = s
            if (!(groupid contains s)){
              for(id <- groupid){
                if(PenFunc(x=groupcenter(id.toInt)-s, penalty=penalty.value, gamma=gamma.value,
                  lambda=lambda/50, a=c, lassophi=lassophi.value) == 0.0){
                  groupres = id
                }
              }
            }
            groupres
          })
          // 更新新组别的均值
          var thisgroup = grouphat.map(s=> {if (s == groupid.last) 1.0 else 0})
          groupcenter(groupid.last.toInt) = (thisgroup :* muhat).sum / thisgroup.sum
          // 查找是否还有新组别
          newcenter = false
          val loop = new Breaks
          loop.breakable {
            for( a <- grouphat.toDenseVector){
              if(!(groupid contains a)){
                groupid += (groupid.last + 1)
                groupcenter += a
                newcenter = true
                loop.break
              }
            }
          }
        }
//        println("finalgroupnum: ")
//        print(groupid.length)
//        println("finalgroup:")
//        println(grouphat(::,0))
//        println(grouphat(::,1))
//        println(grouphat(::,2))
        val groupedData = SampledData.map(s=>{
          val dat:DenseMatrix[Double] = DenseMatrix.horzcat(s.head, s(1).t, grouphat(::, s(3)(0,0).toInt).toDenseMatrix.t)
          dat(*, ::).map(ss=>(ss(p+1), (new DenseMatrix[Double](rows = 1, cols = p, ss.toArray.take(p)), Array(ss(p))))).toArray
        }).flatMap(s=>s).reduceByKey((x,y)=>(DenseMatrix.vertcat(x._1, y._1), x._2 ++ y._2))
        // 这里mu由于样本原因可能在小组别里面估计不准确，同时由于抽样问题，均值得到mu不太行，所以后面用预测过的beta再求一下
        val mubetares = groupedData.sortByKey().map(dat=>{
          val mu_thisgroup = dat._2._2.sum / dat._2._1.rows
          // 避免singular
          var beta_thisgroup = betahat
          try {
            beta_thisgroup = (inv(dat._2._1.t * dat._2._1) * dat._2._1.t * (new DenseMatrix[Double](rows=dat._2._1.rows, cols = 1, dat._2._2)-mu_thisgroup)).toDenseVector
          } catch {
            case e: MatrixSingularException => println("Got MatrixSingularException during beta estimation.")
          }
          (mu_thisgroup, beta_thisgroup, dat._2._1.rows.toDouble)
        }).collect()
        betahat = DenseVector.zeros[Double](p)
        println("groupnum: ")
        for(resgroup <- mubetares){
          betahat = betahat + resgroup._2 * resgroup._3
          println(resgroup._3)
        }
        betahat = betahat / N.toDouble
        // 用得到的beta重新预测一下mu，再对beta二次预测，因为发现mseu很小但是由于mseb大所以bic分对组也大
        val uniquemu = groupedData.sortByKey().map(dat=> {
          (new DenseVector[Double](dat._2._2) - dat._2._1 * betahat).toArray.sum / dat._2._1.rows.toDouble
        }).collect()
        muhat = grouphat.map(s=>uniquemu(s.toInt))
        betahat = SampledData.map(s=>{
          inv(s.head.t * s.head)*s.head.t*(s(1).toDenseVector-muhat(::,s(3)(0,0).toInt))
        }).reduce(_+_) * (1/M.toDouble)
//        print("betahat:")
//        println(betahat)
        println("muunique:" + uniquemu.mkString("\t"))
        System.out.println("分组程序运行时间： " + (System.nanoTime - startTime1)/1e9d + "s")
        //BIC
        val bic_p1 = sqrt(SampledData.map(s => {
          val xm = s.head
          val ym = s(1).toDenseVector
          val m = s(3)(0, 0).toInt
          //Sum of l1 loss
          (ym - xm * betahat - muhat(::,m)).map(d=>d*d).sum
        }).sum) / N
        val BIC = bicphi.value*log(log(N)) * (log(N) / N) * uniquemu.length + log(bic_p1)
        println(bicphi.value*log(log(N)) * (log(N) / N) * uniquemu.length)
        println(log(bic_p1))
        println("BIC:" + BIC)
        bic_lambda_mse_k(bic_lambda_mat_index, 0) = BIC
        //MSE
        val MSEB = (betahat - Beta).map(s => abs(s)).sum / p.toDouble
        val MSEU = (muhat - Mu).map(diff => abs(diff)).sum / N.toDouble
        println("MSEB & MSEU:" + MSEB + "," + MSEU)
        bic_lambda_mse_k(bic_lambda_mat_index, 2) = MSEB
        bic_lambda_mse_k(bic_lambda_mat_index, 3) = MSEU
        bic_lambda_mse_k(bic_lambda_mat_index, 4) = muhat.toDenseVector.toArray.distinct.length.toDouble
        beta_of_each_lambda = beta_of_each_lambda :+ betahat.toDenseMatrix
        u_distinct_of_each_lambda = u_distinct_of_each_lambda :+ muhat.toDenseVector.toArray.distinct
        writer.println(lambda + "," + BIC + "," + MSEB + "," + MSEU + "," + bic_lambda_mse_k(bic_lambda_mat_index, 4))
        bic_lambda_mat_index += 1
        println("-----------------------------------------------------------------------------------------------------")
        //println("On lambda" + lambda + "BIC"+BIC+"MSEB"+MSEB+"MSEU"+MSEU+"," + "Beta"+BetaHat.value.toArray.mkString("\t") + "," +"U_distinct"+ u_distinctHat.value.mkString("\t"))
      }
      mse_time_sim += (System.nanoTime - startTime)/1e9d
      System.out.println("分组程序运行时间： " + mse_time_sim.last + "s")
      val BICRes = bic_lambda_mse_k(::, 0).toArray
      val FinalResIndex = BICRes.indexOf(BICRes.min)
      mse_beta_sim += bic_lambda_mse_k(FinalResIndex, 2)
      mse_mu_sim += bic_lambda_mse_k(FinalResIndex, 3)
      mse_k_sim += bic_lambda_mse_k(FinalResIndex, 4)
      mse_bic_sim += bic_lambda_mse_k(FinalResIndex, 0)
      writer.println("LambdaSelect"+","+bic_lambda_mse_k(FinalResIndex, 1)+","+",")
      writer.println("Beta"+ beta_of_each_lambda(FinalResIndex+1)(::,0).toArray.mkString("\t"))
      writer.println("U_distinct"+ u_distinct_of_each_lambda(FinalResIndex+1).mkString("\t"))
      writer.println("Time_total"+","+mse_time_sim.last+","+",")
      println("BETA:" + beta_of_each_lambda(FinalResIndex+1)(::,0).toArray.mkString("\t"))
      println("U_DISTINCT:" + u_distinct_of_each_lambda(FinalResIndex+1).mkString("\t"))
      println("======================================================================================================")
    }
    println("The Final Result is follow:\n" +  "BIC_mean:" + mean(mse_bic_sim) + "BIC_std:" + variance(mse_bic_sim) +
      "MSEB_mean:" + mean(mse_beta_sim) + "MSEB_std:" + variance(mse_beta_sim) +
      "MSEU_mean:" + mean(mse_mu_sim) + "MSEU_std:" + variance(mse_mu_sim) +
      "K_mean:" + mean(mse_k_sim) + "K_std:" + variance(mse_k_sim) +
      "time_mean:" + mean(mse_time_sim) + "time_std:" + variance(mse_time_sim) +
      "time_eachlambda_mean:" + mean(mse_time_sim.map(s=>s/Iter.toDouble)) + "time_eachlambda_std:" + variance(mse_time_sim.map(s=>s/Iter.toDouble)))
    writer.println("The Final Result is follow:\n" +  "BIC_mean:" + mean(mse_bic_sim) + "BIC_std:" + variance(mse_bic_sim) +
      "MSEB_mean:" + mean(mse_beta_sim) + "MSEB_std:" + variance(mse_beta_sim) +
      "MSEU_mean:" + mean(mse_mu_sim) + "MSEU_std:" + variance(mse_mu_sim) +
      "K_mean:" + mean(mse_k_sim) + "K_std:" + variance(mse_k_sim) +
      "time_mean:" + mean(mse_time_sim) + "time_std:" + variance(mse_time_sim) +
      "time_eachlambda_mean:" + mean(mse_time_sim.map(s=>s/lambdanum.toDouble)) + "time_eachlambda_std:" + variance(mse_time_sim.map(s=>s/lambdanum.toDouble)))
    writer.close()
  }
}
