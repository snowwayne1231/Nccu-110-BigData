import requests
import json
import sys
from datetime import datetime
from bs4 import BeautifulSoup
from .refresh_date import formatDate


URL_BITCOIN_WEBSITE = 'https://news.bitcoin.com/page/{}/'
PAGE_MAX = 1585
PAGE_MIN = 1


def crawlBitcoinComWebsites(page, stop_page, old_websites = None):
    all_websites = [] if old_websites is None else old_websites
    page_now = page
    _gap = page_now - stop_page
    _pr = 0
    _i = 0
    _href_map = {}
    for website in all_websites:
        _href_map[website['href']] = True
    

    while (page_now >= stop_page):
        response = requests.get(URL_BITCOIN_WEBSITE.format(page_now))
        soup = BeautifulSoup(response.text, "html.parser")
        div_article = soup.find('div', class_='standard__article')
        storyies = div_article.select('.story--medium')
        
        # print(len(storyies))
        for story in storyies:
            _img = story.find('a', class_='story--medium__img-container')
            _href = _img.get('href')
            _h6 = story.find('h6', class_='story--medium__title')
            _title = _h6.getText().strip()
            _footer = story.find('div', class_='story__footer')
            _text_date = _footer.getText().strip()
            if 'https://news.bitcoin.com/' in _href and _href_map.get(_href,None) is None:
                _splited = _text_date.split('|')
                _type = _splited[0].strip()
                _date = _splited[1].strip()
                _date = formatDate(_date)
                all_websites.append(
                    {'href': _href, 'title': _title, 'date': _date, 'type': _type, 'content': ''}
                )
                _href_map[_href] = True
        page_now -= 1
        _i += 1
        _pr = _i / _gap * 100
        print(' {:2.2f}%'.format(_pr), end='\r')

    print('100.00% - Done')

    return all_websites



def crawlNewsContentByJson(dataset, tmp=False):

    _length_total = len(dataset)
    _i = 0
    _loaded_data = 0
    _loaded_tmp_done = _length_total / 20 if tmp else _length_total
    _now = datetime.now()
    _dt_string = _now.strftime("%Y%m%d_%H%M")
    
    for _ in dataset:
        if _['content'] == '':
            _href = _['href']
            response = requests.get(_href)
            soup = BeautifulSoup(response.text, "html.parser")
            div_article = soup.find('article', class_='article__body')
            article_ps = div_article.find_all('p', limit=3)
            _text = ''
            for _p in article_ps:
                _text += (' ' + _p.getText())

            _['content'] = _text.strip()
            _loaded_data += 1

        _i += 1
        _pr = _i / _length_total * 100
        print(' {:2.2f}%'.format(_pr), end='\r')

        if _loaded_data > _loaded_tmp_done:
            with open('json/news-bitcoin-com-websites-{}.json'.format(_dt_string), 'w') as f:
                json.dump(dataset, f, indent=2)
            _loaded_data = 0

    print('100.00% - Done')

    with open('json/news-bitcoin-com-websites-{}.json'.format(_dt_string), 'w') as f:
        json.dump(dataset, f, indent=2)

    return dataset


if __name__ == '__main__':

    argvs = sys.argv[1:]
    length_argvs = len(argvs)

    if length_argvs == 0:

        websites = crawlBitcoinComWebsites(PAGE_MAX, PAGE_MIN)
        with open('json/news-bitcoin-com-websites.json', 'w') as f:
            json.dump(websites, f, indent=2)
        
    elif length_argvs == 1:

        _json_path = argvs[0]
        _json_data = None

        with open(_json_path, 'r') as f:
            _json_data = json.load(f)

        crawlNewsContentByJson(_json_data, True)

    elif length_argvs == 2:

        _json_path = argvs[0]
        _page_max = int(argvs[1])
        _json_data = None

        with open(_json_path, 'r') as f:
            _json_data = json.load(f)

        next_json = crawlBitcoinComWebsites(_page_max, PAGE_MIN, _json_data)

        crawlNewsContentByJson(next_json, False)



    
