import json, sys
import requests


SLACKAPIURL = 'https://hooks.slack.com/services/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'


def send_msg(msg):
    '''
    
    '''

    dict_headers = {'Content-type': 'application/json'}

    dict_payload = { "text": msg }
    json_payload = json.dumps(dict_payload)

    rtn = requests.post(SLACKAPIURL, data=json_payload, headers=dict_headers)

    return rtn



if __name__ == '__main__':

    msg = 'Test Slack Notification.'
    argvs = sys.argv[1:]
    if len(argvs) == 1:
        msg = argvs[0].strip()
    
    send_msg(msg)