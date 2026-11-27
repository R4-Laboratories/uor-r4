# 'weak' ties 'both' on all science metrics, and wins on fewer factors.
# 'balanced' scores better in development but loses one original exact pair.
def config($train;$dev;$exact;$protect):
 {train:{contrasts:{"0":{directed:$train,exact:0,pairs:192},"1":{pairs:192}}},
  dev:{contrasts:{"0":{directed:$dev,exact:$exact,pairs:200},"1":{pairs:100}}},
  protect:{overall:{pairs:50,exact:$protect},contrasts:{"1":{pairs:50}}}};
{baseline:config(10;10;10;50),
 balanced:config(20;30;30;49),
 weak:config(20;20;20;50),
 both:config(20;20;20;50)}
