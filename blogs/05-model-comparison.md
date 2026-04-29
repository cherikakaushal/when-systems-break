# Do different models break differently?

## Problem

Do all models fail the same way when data becomes imperfect?

## What I tested

- Logistic Regression
- Random Forest

Under:
- clean data
- noisy data
- missing data

## What I observed

- Some models handled noise better than others  
- Tree-based models were more robust to missing values  
- Linear models were more sensitive to data quality  

## Insight

Model choice matters not just for performance, but for reliability under imperfect conditions.

## Conclusion

Understanding how models fail is as important as how they perform.