<h1>Retail Association Mining Using Instacart Dataset</h1>

## 1. Project Overview

This project analyzes historical grocery purchase transactions from the Instacart dataset to identify frequently purchased products and discover associations between items bought together. The analysis combines Exploratory Data Analysis (EDA), the Apriori algorithm, association rule mining, and the Chi-Square Test of Independence.

The objective is to understand customer purchasing patterns and identify potential product bundling and recommendation opportunities using transaction data.

## 2. Objectives

- Analyze grocery purchasing patterns using historical transaction data.
- Clean and preprocess product and order information.
- Perform Exploratory Data Analysis to understand basket sizes, popular products, departments, and aisles.
- Apply the Apriori algorithm to identify frequent product combinations.
- Generate association rules using support, confidence, and lift.
- Apply the Chi-Square Test of Independence to examine the statistical association between selected products.
- Present the results using tables, charts, and statistical summaries.

## 3. Dataset

The project uses the Instacart Market Basket Analysis dataset.

The following files are used:

link :https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis/data

| File | Description |
|---|---|
| `orders.csv` | Contains order information, customer identifiers, order sequence, day, and hour. |
| `order_products__prior.csv` | Contains products purchased in previous orders and reorder indicators. |
| `products(1).csv` | Maps product IDs to product names and category identifiers. |
| `aisles.csv` | Maps aisle IDs to aisle names. |
| `departments.csv` | Maps department IDs to department names. |

The prior-order transaction data provides the basis for basket construction and association analysis.

## 4. Technologies and Libraries

- **Python** — implementation and analysis.
- **Pandas** — data loading, cleaning, grouping, merging, and tabular analysis.
- **NumPy** — numerical operations and matrix calculations.
- **SciPy** — sparse matrix operations and statistical hypothesis testing.
- **Matplotlib** — visualization of distributions, product frequencies, departments, and association-rule lift.

## 5. Methodology

### Step 1: Data Loading

The required CSV and ZIP files are loaded using Pandas. The script supports the prior-order dataset in either compressed ZIP or uncompressed CSV format.

### Step 2: Data Cleaning and Integration

The product catalog is cleaned by removing records with missing product identifiers or names and removing duplicate product IDs. Product names are trimmed to eliminate unnecessary whitespace. Duplicate keys are removed from aisle and department lookup tables.

The product catalog is then merged with aisle and department information to provide descriptive labels for subsequent analysis. The original dataset files remain unchanged.

### Step 3: Transaction Preprocessing

Transactions are represented as baskets of products, with each basket corresponding to an order.

The script processes the prior-order dataset in chunks to reduce memory usage. Duplicate order-product combinations are removed, basket sizes are calculated, and product purchase frequencies are aggregated.

A binary sparse matrix is constructed for selected products. Each entry indicates whether a product appeared in an order, regardless of the quantity purchased.

### Step 4: Exploratory Data Analysis (EDA)

The analysis examines:

- Total prior orders and transaction item lines.
- Number of unique customers and catalog products.
- Mean and median basket sizes.
- Percentage of reordered item lines.
- Average order sequence and days between orders.
- Busiest day and hour of ordering.
- Most frequently purchased products.
- Product distribution across departments and aisles.

Matplotlib is used to generate charts that summarize basket-size distributions, leading departments, and popular products.

### Step 5: Apriori Algorithm

The Apriori algorithm identifies frequent product itemsets using a minimum support threshold of **0.25%**.

The implementation first identifies frequent individual products and then evaluates pairs among the 60 most frequently purchased products. A pair is considered frequent when its support meets the specified minimum threshold.

Apriori uses the downward-closure property: if an itemset is frequent, its subsets must also be frequent. This property helps reduce the number of candidate combinations considered.

### Step 6: Association Rule Mining

Frequent product pairs are converted into directional association rules. The following metrics are calculated:

**Support**

The proportion of all orders containing both the antecedent and consequent.

Support(A → B) = Orders containing both A and B / Total orders

**Confidence**

The proportion of orders containing A that also contain B.

Confidence(A → B) = Orders containing both A and B / Orders containing A

**Lift**

The ratio of the observed confidence to the expected confidence if the two products were independent.

Lift(A → B) = Confidence(A → B) / Support(B)

A lift greater than 1 indicates a positive association between the products. Rules are filtered using a minimum confidence threshold of **10%** and ranked by lift.

### Step 7: Confirmatory Association Analysis

Selected product pairs from the report are evaluated using their order-level occurrence counts. This provides support, confidence, and lift values for the specified pairs using the complete prior-order basket population.

The selected associations include:

- Organic Ginger Root and Organic Garlic.
- Organic Cilantro and Limes.
- Organic Garlic and Organic Yellow Onion.
- Raspberries and Strawberries.
- Limes and Large Lemon.
- Organic Cilantro and Organic Red Onion.

### Step 8: Chi-Square Test of Independence

The Chi-Square Test of Independence is used to examine whether purchases of Organic Ginger Root and Organic Garlic are statistically associated.

A 2 × 2 contingency table is constructed using orders in which both products are present, only one product is present, or neither product is present.

**Null hypothesis (H₀):** The presence of Organic Ginger Root and Organic Garlic in an order is independent.

**Alternative hypothesis (H₁):** The presence of Organic Ginger Root and Organic Garlic in an order is associated.

**Significance level:** α = 0.05

The test statistic is calculated as:

χ² = Σ (O − E)² / E

where O represents the observed frequency and E represents the expected frequency under the null hypothesis.

The degrees of freedom for a contingency table are calculated as:

df = (number of rows − 1) × (number of columns − 1)

The p-value is compared with the significance level of 0.05. If p < 0.05, the null hypothesis is rejected; otherwise, it is not rejected. The analysis also calculates the critical chi-square value for comparison.

## 6. Results and Interpretation

The analysis examines a large collection of historical grocery transactions and identifies recurring purchasing patterns.

The descriptive analysis highlights popular products, basket sizes, reorder behavior, and product categories with high transaction frequencies. Association rules reveal product combinations that occur more often than expected based on individual product frequencies.

The Chi-Square Test of Independence provides a statistical assessment of the relationship between Organic Ginger Root and Organic Garlic purchases.

Association measures describe patterns in the observed data; they do not establish that purchasing one product causes a customer to purchase another.

## 7. Output Files

After execution, the script creates an `analysis_outputs/` directory containing the following results.

### Data and Analysis Tables

- `cleaned_product_catalog.csv`
- `top_products.csv`
- `department_counts.csv`
- `aisle_counts.csv`
- `apriori_frequent_1_itemsets.csv`
- `apriori_frequent_2_itemsets.csv`
- `apriori_rules_top60.csv`
- `report_rules_verified.csv`
- `chi_square_observed.csv`
- `chi_square_expected.csv`

### Visualizations

- `basket_size_histogram.png`
- `department_chart.png`
- `top_products_chart.png`
- `selected_rule_lift.png`

## 8. Installation and Execution

### Prerequisites

Install Python and the required libraries:

```bash
pip install pandas numpy scipy matplotlib
```

### Run the Analysis

Execute the following command from the project directory:

```bash
python Team_11.py
```

The analysis results, tables, figures, and statistical summary are written to the `analysis_outputs/` directory.

## 9. Conclusion

This project demonstrates a transaction-analysis workflow that combines data preprocessing, exploratory analysis, frequent-itemset mining, association-rule generation, and statistical hypothesis testing.

The Apriori-based analysis identifies recurring product combinations, while support, confidence, and lift quantify their strength. The Chi-Square Test of Independence provides an additional statistical method for evaluating a selected product association. Together, these techniques can support exploratory market-basket analysis, product bundling decisions, and recommendation-system development.

