from curses.ascii import isdigit

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# %% load the data
spark = SparkSession.builder\
    .master("local")\
    .appName("adult")\
    .getOrCreate()

fname = "./adult/adult.data"
file = open(fname)

# clean up the data
lines = file.read().splitlines()
cleaned = []
bad = []
for line in lines:
    if line.count(',') == 14:
        line = line.split(', ')
        line = [int(item) if item.isdigit() else item for item in line]
        cleaned.append(line)
    else:
        bad.append(line)

schema = StructType([
    StructField('age', IntegerType(), True),
    StructField('workclass', StringType(), True),
    StructField('fnlwgt', IntegerType(), True),
    StructField('education', StringType(), True),
    StructField('education_num', IntegerType(), True),
    StructField('marital_status', StringType(), True),
    StructField('occupation', StringType(), True),
    StructField('relationship', StringType(), True),
    StructField('race', StringType(), True),
    StructField('sex', StringType(), True),
    StructField('capital_gain', IntegerType(), True),
    StructField('capital_loss', IntegerType(), True),
    StructField('hours_per_week', IntegerType(), True),
    StructField('native_country', StringType(), True),
    StructField('income', StringType(), True)
])

# Load the data and split by comma
adult_rdd = spark.sparkContext.parallelize(cleaned)

# Create DataFrame from RDD with specified column names
adult = spark.createDataFrame(adult_rdd, schema=schema)

# %% ratio of man for each marital status
man = adult.where(col('sex') == 'Male').groupBy('marital_status').count()
tot = adult.groupBy('marital_status').count()
print(f'Man for each marital status')
man.show()
print(f'Total people for each marital status')
tot.show()
man = (man.withColumnRenamed('marital_status','man_marital_status')
       .withColumnRenamed('count','man_count'))
ratios = man.join(tot, man.man_marital_status == tot.marital_status, how='left')
ratios = ratios.withColumn('ratio', col('man_count') / col('count')).select('marital_status', 'ratio')
print(f'Man ratio for each marital status')
ratios.show()

# %% average work hour of female who earns >50K for each country
avg_work_hour = adult.where(col('income') == '>50K').groupBy('native_country').avg('hours_per_week')
avg_work_hour.show()

# %% min and max education level for ppl. who earn different money
income_groups = adult.groupBy('income')
income_groups_max_edu_level = (income_groups
                                .max('education_num'))
income_groups_min_edu_level = (income_groups
                                .min('education_num'))
income_groups_max_edu_level = (income_groups_max_edu_level
                               .join(adult, col('max(education_num)') == col('education_num'), how='left')
                               .select(income_groups_max_edu_level.income, 'education')
                               .withColumn('lowest_education', lit(None).cast(StringType()))
                               .withColumnRenamed('education', 'highest_education'))
max_edu_income_g50K  = income_groups_max_edu_level.where(col('income') == '>50K').take(1)
max_edu_income_le50K = income_groups_max_edu_level.where(col('income') == '<=50K').take(1)
income_groups_min_edu_level = (income_groups_min_edu_level
                               .join(adult, col('min(education_num)') == col('education_num'), how='left')
                               .select(income_groups_min_edu_level.income, 'education')
                               .withColumn('highest_education', lit(None).cast(StringType()))
                               .withColumnRenamed('education', 'lowest_education')
                               .select('income', 'highest_education', 'lowest_education'))
min_edu_income_g50K  = income_groups_min_edu_level.where(col('income') == '>50K').take(1)
min_edu_income_le50K = income_groups_min_edu_level.where(col('income') == '<=50K').take(1)
income_edu_tbl = spark.createDataFrame([
    min_edu_income_le50K[0], min_edu_income_g50K[0],
    max_edu_income_le50K[0], max_edu_income_g50K[0],
])
income_edu_tbl = (income_edu_tbl
                    .join(income_edu_tbl
                        .withColumnsRenamed({
                            'income': 'income1',
                            'highest_education': 'highest_education1',
                            'lowest_education': 'lowest_education1',
                        }))
                    .where((col('income') == col('income1')) & col('highest_education').isNotNull() & col('lowest_education1').isNotNull())
                    .select('income', 'highest_education', 'lowest_education1')
                    .withColumnRenamed('lowest_education1', 'lowest_education'))
income_edu_tbl.show()
