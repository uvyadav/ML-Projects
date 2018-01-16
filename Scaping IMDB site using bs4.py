import re
from bs4 import BeautifulSoup
from urllib.request import urlopen as uReq

my_url = "http://www.imdb.com/list/ls058293270/"

url = my_url

filename = "Movie_dataset.csv"
f = open(filename, "a")
headers = "Movie_name, Year, Rating\n"
f.write(headers)

for i in range(4):
    uClient = uReq(url)
    page = uClient.read()
    uClient.close()
    
    soup= BeautifulSoup(page, "html.parser")
    soup.select_one()
    zz= soup.findAll("div",attrs={"class":"info"})
    """
    for x in range(len(zz)):
        movie_name = zz[x].b.a.text
        year = zz[x].b.span.text[1:5]
        r=zz[x].findAll("span",{"class":"value"})
        rating = r[0].text
        
        f.write(movie_name.replace("," , ".") + "," + year + "," + rating + "\n")
     """   
    next_page = soup.findAll("div",attrs={"class":"pagination"}, recursive = True) 
    print (next_page)
    url_next = next_page[0].a['href']
    url = my_url+url_next
f.close()


"""
if "Next" in next_page[0].a.text:
    print("yes")
else:
    print("Ohh yes")
   """
   
"""with open ("movie.csv",'a') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow([movie_name,year,rating]) """