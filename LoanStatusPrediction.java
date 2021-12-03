package org.ml;
import java.io.IOException;

import tech.tablesaw.api.Table;
import tech.tablesaw.plotly.Plot;
import tech.tablesaw.plotly.components.Figure;
import tech.tablesaw.plotly.components.Layout;
import tech.tablesaw.plotly.traces.HistogramTrace;

import java.util.Arrays;

import weka.classifiers.Classifier;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class LoanStatusPrediction {
	public static Instances getInstances (String filename)
	{
		
		DataSource source;
		Instances dataset = null;
		try {
			source = new DataSource(filename);
			dataset = source.getDataSet();
			dataset.setClassIndex(dataset.numAttributes()-1);
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
			
		}
		
		return dataset;
	}
	
	public static void main(String args[]) {
		System.out.println("Loan Status Prediction");
		try {
			Table bank_data = Table.read().csv("C:\\Users\\reddy\\eclipse-workspace\\org.ml\\src\\main\\java\\org\\ml\\train_dataset.csv");
			System.out.println(bank_data.shape());
			
			System.out.println(bank_data.structure());
			System.out.println(bank_data.first(5));
			
			Layout layout1 = Layout.builder().title("Distribution of ApplicantIncome").build();
			HistogramTrace trace1 = HistogramTrace.builder(bank_data.nCol("ApplicantIncome")).build();
			Plot.show(new Figure(layout1, trace1));
		
			Layout layout2 = Layout.builder().title("Distribution of Credit Histroy").build();
			HistogramTrace trace2 = HistogramTrace.builder(bank_data.nCol("Credit_History")).build();
			Plot.show(new Figure(layout2, trace2));
		
			Layout layout3 = Layout.builder().title("Distribution of LoanAmount").build();
			HistogramTrace trace3 = HistogramTrace.builder(bank_data.nCol("LoanAmount")).build();
			Plot.show(new Figure(layout3, trace3));
			
			Layout layout4 = Layout.builder().title("Distribution of Married Status").build();
			HistogramTrace trace4 = HistogramTrace.builder(bank_data.nCol("Married")).build();
			Plot.show(new Figure(layout4, trace4));
			
			Instances train_data = getInstances("C:\\\\Users\\\\reddy\\\\eclipse-workspace\\\\org.ml\\\\src\\\\main\\\\java\\\\org\\\\ml\\\\train_Weka_dataset.arff");
			Instances test_data = getInstances("C:\\\\Users\\\\reddy\\\\eclipse-workspace\\\\org.ml\\\\src\\\\main\\\\java\\\\org\\\\ml\\\\test_sample.arff");
			System.out.println(train_data.size());
			
			/** Classifier here is Linear Regression */
			Classifier classifier1 = new weka.classifiers.functions.Logistic();
			/** */
			classifier1.buildClassifier(train_data);
			
			
			/**
			 * train the alogorithm with the training data and evaluate the
			 * algorithm with testing data
			 */
			Evaluation eval1 = new Evaluation(train_data);
			eval1.evaluateModel(classifier1, test_data);
			/** Print the algorithm summary */
			System.out.println("** Logistic Regression Evaluation with Datasets **");
			System.out.println(eval1.toSummaryString());
//			System.out.print(" the expression for the input data as per alogorithm is ");
//			System.out.println(classifier);
			
			double confusion[][] = eval1.confusionMatrix();
			System.out.println("Confusion matrix:");
			for (double[] row : confusion)
				System.out.println(	 Arrays.toString(row));
			System.out.println("-------------------");

			System.out.println("Area under the curve");
			System.out.println( eval1.areaUnderROC(0));
			System.out.println("-------------------");
			
			System.out.println(eval1.getAllEvaluationMetricNames());
			
			System.out.print("Recall :");
			System.out.println(Math.round(eval1.recall(1)*100.0)/100.0);
			
			System.out.print("Precision:");
			System.out.println(Math.round(eval1.precision(1)*100.0)/100.0);
			System.out.print("F1 score:");
			System.out.println(Math.round(eval1.fMeasure(1)*100.0)/100.0);
			
			System.out.print("Accuracy:");
			double acc = eval1.correct()/(eval1.correct()+ eval1.incorrect());
			System.out.println(Math.round(acc*100.0)/100.0);
			
			
			System.out.println("-------------------");
//			Instance predicationDataSet = test_data.get(2);
//			double value = classifier1.classifyInstance(predicationDataSet);
			/** Prediction Output */
//			System.out.println("Predicted label:");
//			System.out.print(value);
			
			}
		catch (IOException e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
		
		
		
		
		
	}
	

}
