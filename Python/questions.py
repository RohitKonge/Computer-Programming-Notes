import requests, bs4

i = 1
authors = set()
while True:
    try:
            result = requests.get(("http://quotes.toscrape.com/page/{}/").format(i))

            soup = bs4.BeautifulSoup(result.text, "lxml")

            for k in soup.select(".author"):
                authors.add(k.text)
                
            for j in authors:
                print(j)
                
            i = i + 1
        
    except:
        print(result.text)
        print("asdasf")
        break
    
