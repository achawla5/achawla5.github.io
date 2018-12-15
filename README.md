CMSC320 Final Project

Members - Akash Chawla, Philip Lo, Kishan Segu

### Overview

Music has become part of many people’s lifestyles over time. It is a form of both personal expression and a kind of social interaction. Music consumption has increased tremendously over the past few years, and a number of companies such as Spotify, Gaana.com, Saavn, Pandora etc. have emerged providing streaming services. With the increasing availability of these streaming services and Bluetooth headsets, we have started to spend more time finding and adding songs to our playlists. By recommending users the music they like, companies are able to lure more customers and thus generate more revenue. With this change in the industry, music recommendations becomes increasingly important for customers in order to lure them into their streaming service. In this notebook, we will analyze and explore the Million Song Dataset, and find relationships between various features. Ultimately, we will also look into recommender systems, and how they're personalized for different individuals based on past history, or finding similar users in such a way that they like similar music.


The dataset we are using is the Million song dataset, which is a freely-available collection of audio features and metadata for a million contemporary popular music tracks. The dataset was taken from:

https://labrosa.ee.columbia.edu/millionsong/

Since this dataset is approximately 280GB in size, we'll be working with a subset of data in order to conduct our analysis. 

## Importing Libraries

- Numpy is imported since pandas relies on numpy to function
- Pandas is imported because of its ease of use. In this project we don't have any SQL files nor do we need to perform any SQL database related commands. In the case we do need SQL, pandas also supports that thus it's the best option for us
- Matplotlib is imported for graphing. It's heavily used in the data exploration section to get some visualization of the data. The %matplotlib inline command tells matplotlib to present graphs in order with the code in the cells
- Seaborn is used in the same manner as matplotlib but for more advanced visualization such as heatmaps
- Sklearn is imported for use in the ML section of this tutorial. The KMeans library was specifically imported to partition our data into clusters based on the features we choose 

Furthermore, we would recommend using Jupyter Notebook for this tutorial since Python is already included, and importing all the necessary libraries is very easy.

## Reading in Dataset

The Million Songs Dataset we have is a csv file so we read it in with pandas. The datafile is relatively large so we will look at the first 20 rows rather than the default first 5 given by the head() function. Taking an initial look at the data file we see that there are some immediate issues with the dataset. There is an unnamed column with entries that correspond to the index of each row. There is also some missing data in artist.playmeid with with multiple entries having an id of -1. In the year column, there are multiple rows with entries of 0 all of which are missing data. The songs are identified by song.id rather than song name.

Here's a link where you can go to further understand the dataset, and how different fields are calculated:

https://labrosa.ee.columbia.edu/millionsong/pages/field-list

## Data Cleaning

Cleaning up the dataset is very important prior to any analysis. 
We know that the unnamed column corresponds to indexing so we can drop that column, then check for any columns that have all data missing. We do this by writing a function that iterates through each row of each column and checks for repeating values.

For our dataset, both energy and danceability have missing values so we'll drop them from our dataset. Additionally we do some further processing on our dataset for better readability.



