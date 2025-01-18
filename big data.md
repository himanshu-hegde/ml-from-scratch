Big Data Analytics HW1
Name: Himanshu Hegde
SUID: 406632168


1)	Pick an industry or sector of the economy that interests you and explain why you chose it. This could be an area that you want to work in someday, or one in which you already have experience. It could even be a specific organization. It must be somewhere that uses Big Data for analytics.

Having worked in the Consumer-Packaged Goods (CPG) / Fast-Moving Consumer Goods (FMCG) space and having built multiple solutions leveraging Data Engineering and Data Science across the entire life-cycle of a CPG product:  Raw material Procurement ->Payments -> Contracts -> Supply chain and logistics -> Warehouse/store/e-comm inventory -> Marketing and Demand generation, I have seen the impact that Big Data Management techniques brings in streamlining the entire process. This is why I am selecting the CPG/FMCG domain for the purpose of this assignment.

2)	Describe how Big Data is used in this area, generally. This could include predicting sales in the retail industry, predicting disease patterns in healthcare or disease outbreaks in public health, identifying fraud in banking, facilitating the use of algorithmic trading in finance, microtargeting in political campaigns, or tailoring personalized recommendations in the entertainment industry. Be sure to fully explain the importance of this use case to the industry or organization, and how it impacts operations and decision-making.

Let us look into a problems being solved in the CPG space:
a)	Raw Material Procurement:
Most manufacturers of CPG products source their Raw Materials from different vendors, under different contractual obligations. This leads to a lot of data being generated, which if used correctly can be used to generate a lot of insights. A few examples could be:
•	Supplier risk assessment and performance analysis- Analyzing supplier performance in terms of timely delivery, quality of products delivered, defective products etc. is essential to the decision-making process. This requires a lot of data regarding the suppliers to be processed at scale.
•	Demand-driven procurement optimization - This can be used to predict commodity price based on historical data. Data for this includes prices of raw materials, transportation costs, global demand, etc, which could run into millions of records.

b) Payments:
A CPG company processing millions of transactions daily. This leads to a lot of data being generated, such as vendor payment data,customer payment data, supplier payment data etc. , which if used correctly can be used to generate a lot of insights. Some examples could be:
•	Real-time fraud detection in transactions - Use anomaly detection algorithms to flag suspicious activities, potentially preventing significant financial losses
•	Working capital optimization - Predictive analytics can optimize cash flow by forecasting payment timings and suggesting ideal payment schedules to suppliers.
c) Contracts:
Many CPG products have contractual obligations. Contracts between Vendors and the company are used to manage the payment process,define scope of work and services to be provided, dismute management framework and terms.This leads to a lot of data being generated, which if used correctly can be used to generate a lot of insights. Some examples could be:
•	Contractual risk management - Contractual risk management is a critical component of payment management. Contractual risk management can be used to monitor and manage risk in the payment process.
•	AI-powered contract analysis and risk assessment - These technologies can analyze thousands of contracts rapidly, extracting key terms, identifying potential risks, and ensuring compliance.
•	Compliance monitoring and alerting - Monitoring and alerting can help ensure compliance with contractual obligations.
d) Supply Chain + Logistics:
•	Route optimization and real-time fleet management- Optimizing routes and fleet management is essential for cost savings and efficiency. This required geospatial data to be processed at scale, using various route algorithms, constraints ad optimization techniques.
•	Predictive maintenance for manufacturing equipment - Using machine learning algorithms to predict equipment failures based on historical data and real-time sensor data.

e) Store + E-commerce:

•	Omnichannel inventory optimization - Omni refers to a combination of multiple channels (offline and online). Omnichannel inventory optimization can be used to optimize inventory placement across multiple channels, using historical customer and product interatcion data.
•	In-store analytics (heat mapping, shopper behavior analysis) - Effective product placement drives revenue. Collaborative filtering and content-based algorithms can be used to optimize inventory placement, using historical customer and product interatcion data.

f) Marketing and Demand Creation:
Product demand creatrion is essential for the success of any business. E-comm platforms now enable hyper-personalization and precise targeting of products. 
•	Customer segmentation and personalized marketing - By analyzing vast amounts of customer data, including demographics, purchase history, online behavior, and even location data, companies can create highly targeted marketing campaigns that are tailored to individual customers.
•	Marketing mix modeling and attribution analysis - Evaluating marketing campaigns across various verticals and channels can be used to understand the effectiveness of marketing campaigns, and re-calibrate strategies. The data included in this scope would be revenue across different channels and verticals, media impressions, clicks, and conversions.



3) Discuss the data being used — its variety, sources, scale, and how frequently it is updated. Briefly touch on how data collection, storage, analysis, and interpretation are carried out, along with the specific tools, technologies, or software platforms utilized.

As mentioned above, a huge amount of data is being generated from the CPG space. It is important to have a clear understanding of the data being used.Data collection, storage, analysis, and interpretation are carried out, along with the specific tools, technologies, or software platforms utilized. 

Data could be structured, semi-structured, or unstructured. It could also be real-time or historical/periodic.

Structured data could be of the following types:

•	Vendor data
•	Customer data
•	Supplier data
•	Contractual data
•	Product data
•	Transaction data
•	Media data
•	Marketing data
•	Store data
•	Financial data
•	Marketing mix data
•	Inventory data

This is usually collected using Point-of-sale (POS) systems and e-commerce platforms

It could also be collected from different vendors and third party sources specilizing in data collection (market research companies such as Nielson, etc).

For data storage, CPG companies typically employ a multi-layered approach:

• Cloud platforms (e.g., AWS, Azure, Google Cloud) for scalable storage and processing
• Data lakes for storing vast amounts of raw data
• Data warehouses for structured, analyzable data

Specific technologies often include:

•Hadoop and Spark for big data processing
•Kafka or Apache Flink for real-time data streaming
