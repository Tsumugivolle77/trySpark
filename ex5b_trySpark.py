from pyspark import SparkConf, SparkContext
from string import punctuation

'''
Used transformations | actions
`sortBy`             | `reduceByKey`
`flatMap`            | `take`
`count`              | `collect`
'''

conf = SparkConf().setAppName("Spark App").setMaster("local")
sc = SparkContext(conf=conf)
fname = "lorem.txt"
text = sc.textFile(fname).flatMap(lambda line: line.translate(str.maketrans('', '', punctuation)).split(" "))
totCount = text.count()
counted = text.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
sortByOccurences = counted.sortBy(lambda x: x[1], ascending=False)
top5MostUsedWords = sortByOccurences.take(5)
print(f"There are in total {totCount} words in the text, where\ntop 5 most used words are: {top5MostUsedWords}")

'''
The sample input is given below:

Lorem ipsum dolor sit amet, consectetur adipiscing elit. Duis sit amet tortor placerat odio mattis mollis vitae ac metus. Pellentesque vitae ex at lacus aliquet vulputate. Nullam dapibus felis ac mattis malesuada. Aliquam viverra at ex vestibulum vehicula. Proin vel leo ut nibh molestie fringilla. Quisque egestas vitae erat vulputate tristique. Curabitur elementum mi non porttitor consequat. Suspendisse et augue neque. Nunc at magna viverra mauris pellentesque maximus in ac lectus. Vivamus sagittis semper enim, et pretium mauris elementum non. Fusce elementum ornare risus, ut consectetur orci sagittis in. Vivamus ac finibus erat. Sed eu mollis augue. Suspendisse dui ex, condimentum eget mi quis, dignissim sollicitudin ante. Proin congue ornare diam at euismod. Ut sit amet est a nunc iaculis dapibus quis et quam.

Curabitur ante arcu, tincidunt in libero in, maximus venenatis turpis. Vivamus tincidunt urna eget ligula mattis tempus. Nunc sapien nisi, condimentum nec lacinia volutpat, tempus sit amet tortor. Integer fringilla vestibulum ipsum, at varius erat tincidunt mattis. Nam ac nulla at neque porttitor mattis vitae at neque. Vivamus venenatis, lectus id gravida tristique, eros tortor faucibus sem, non tincidunt lectus purus id est. Maecenas pretium odio nunc, vitae vestibulum libero convallis ac. Aenean vitae justo maximus, hendrerit velit blandit, pretium nibh. Lorem ipsum dolor sit amet, consectetur adipiscing elit. Aenean pharetra lacus massa, non fermentum sapien aliquam nec. Integer tortor dui, tincidunt pellentesque egestas ut, scelerisque ac eros. Vivamus et justo nulla. Nam eget tellus interdum, hendrerit neque at, volutpat turpis.

Pellentesque aliquam malesuada nulla, non ornare massa porttitor vehicula. Suspendisse bibendum leo quis justo lacinia, non facilisis orci fermentum. Mauris in iaculis justo. Quisque vitae ornare augue. Morbi tincidunt quam massa, eu semper nisl posuere vitae. Quisque nunc eros, auctor et facilisis non, euismod in lacus. In at purus arcu. Vestibulum varius, magna et ullamcorper luctus, mauris mi volutpat felis, et lobortis sapien eros eget nunc. Pellentesque ultrices, metus vel volutpat consequat, leo ipsum efficitur purus, ac dignissim odio eros id mi. Aliquam scelerisque, purus in ullamcorper imperdiet, libero augue blandit nunc, quis placerat mi elit id tellus. Nam nec euismod leo. Pellentesque non malesuada libero. In elementum placerat velit non pharetra. Pellentesque lacinia felis quam, ac auctor risus posuere eu. Donec ullamcorper mollis nulla, at commodo augue luctus ut. Pellentesque facilisis molestie enim id rutrum.

Vestibulum in feugiat massa. Phasellus consectetur ut orci quis vestibulum. Praesent imperdiet ut eros id congue. Nulla facilisi. Vivamus ante tellus, lacinia ut ornare sed, feugiat sit amet magna. Nunc sagittis varius metus, at ultrices tellus tincidunt scelerisque. Phasellus consectetur eu lectus eu efficitur. Proin congue nibh quis elementum aliquet. Nunc tincidunt eu ipsum vel tempus. Nunc at ultrices nulla. Sed leo metus, laoreet et eros sed, iaculis aliquam turpis.

Duis ac purus eleifend, semper metus fringilla, semper nunc. Nullam purus sapien, rhoncus efficitur vehicula et, bibendum sed quam. Sed maximus orci purus, consectetur dapibus dui convallis ut. In maximus venenatis malesuada. Duis commodo, sapien a imperdiet lacinia, lacus elit egestas felis, id commodo elit ante quis tellus. Nullam et mauris vel dui ultrices porttitor luctus ac odio. Praesent sit amet nisl vel orci fringilla dictum. Proin nec viverra eros, quis posuere ex. Ut a lectus tortor. Aliquam erat volutpat. Maecenas est mi, hendrerit quis lacinia in, facilisis commodo leo. Nulla feugiat mi in mi venenatis, ut interdum felis vestibulum. Vestibulum ultrices, eros nec feugiat pulvinar, augue est maximus dui, id faucibus nulla nibh nec elit. Mauris venenatis tortor ut justo facilisis, eu pharetra erat accumsan.
'''