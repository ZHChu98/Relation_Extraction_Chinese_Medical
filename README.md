# **Relation Extraction on Chinese Medical Corpus**

## **Brief Description**

In recent years, people are looking forward to a revolution in the medicine area called "AI+medecine". However, due to the lack of structuralization of data, most of the medical datasets are in the form of natural language. We hope to find an automatic machine learning way to extract semantic relations among medical terms, in order to save manual work on the construction of knowledge base in the medical area.

## **Project Structure**

1. Preprocess the data such as parsing, pos-tagging, using *HanLP*
2. Implement supervised deep netork such as CNN, biLSTM using *PyTorch* on MNIST to test their performace
3. Re-train the models on Medical corpus to extract relationship between entities given in corpus and optimize through adding position indicator, tuning the models

For more information, please view the [report](Report.pdf).
