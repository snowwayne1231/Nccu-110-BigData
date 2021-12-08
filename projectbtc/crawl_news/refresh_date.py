import sys, json
from datetime import datetime, timedelta

def formatDate(datestr):
    if datestr[:4].isnumeric():
        return datestr
    
    try:
        dt = datetime.strptime(datestr, '%b %d, %Y')
    except:
        if 'ago' in datestr:
            _number = int(datestr[:2].strip())
            dt = datetime.today() - timedelta(days=-(_number))
        else:
            print('Exception datestr : ', datestr)
            return datestr

    return dt.strftime('%Y / %m / %d')


if __name__ == '__main__':

    argvs = sys.argv[1:]
    length_argvs = len(argvs)

    if length_argvs == 1:
        _json_data = None
        _json_file_path = argvs[0]
        with open(_json_file_path, 'r') as f:
            _json_data = json.load(f)

        for _ in _json_data:
            _['date'] = formatDate(_['date'])

        with open(_json_file_path, 'w') as f:
            json.dump(_json_data, f, indent=2)
            
            