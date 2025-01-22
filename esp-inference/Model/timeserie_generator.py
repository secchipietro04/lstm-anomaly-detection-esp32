from datetime import timedelta
from mockseries.trend import LinearTrend, FlatTrend
from mockseries.seasonality import SinusoidalSeasonality
from mockseries.utils import datetime_range, plot_timeseries, write_csv
from datetime import datetime
import random
from mockseries.noise import RedNoise
trend = FlatTrend(0)
MU, SIGMA=1, 0.15

seasonality = SinusoidalSeasonality(amplitude=0.0, period=timedelta(seconds=1)) 
for i in range(3, 8):
    seasonality+= SinusoidalSeasonality(amplitude=1/(pow(i,2))*random.gauss(MU,SIGMA)
, period=timedelta(seconds=random.gauss(MU,SIGMA)*3*pow(i,2)))
noise = RedNoise(mean=0, std=0.01*random.gauss(MU,SIGMA)
, correlation=0.5*random.gauss(MU,SIGMA)
)
timeseries = trend + seasonality + noise


time_points = datetime_range(
    granularity=timedelta(seconds=1/10),
    start_time= datetime.now(),
    end_time=datetime.now() + timedelta(hours=1),
)
ts_values = timeseries.generate(time_points=time_points)

plot_timeseries(time_points, ts_values)
timeseries.preview_year()
write_csv(time_points, ts_values, "hello_mockseries.csv")
