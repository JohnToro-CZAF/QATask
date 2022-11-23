import re
import string

def handleReaderAns(answer, question):
    if answer.startswith("."):
        # an email address .google
        answer = answer[1:]
    if answer.endswith(".") or answer.endswith(","):
        # a sentence . or a list of items
        answer = answer[:-1]
    # If answer is a province then there is no ambiguity
    unit_list = ['Tỉnh', 'tỉnh', 'thành phố', 'Thành phố', 'Thành Phố']
    for unit in unit_list:
      if unit in answer: answer = answer.replace(unit, '')
    
    return answer.strip()

def matching(short_form: str, wikipage: str):
    match = 0
    words_wiki = wikipage[5:].replace('_', ' ').split()
    words_short_form = short_form.split()
    decay = 0.5
    constant = 1
    for w in words_short_form:
        if w in words_wiki:
            match += constant
            constant *= decay
    return match

def matching_nospace(short_form: str, wikipage: str):
    match = 0
    words_wiki = wikipage.split()
    words_short_form = short_form.split()
    decay = 0.5
    constant = 1
    for w in words_short_form:
        if w in words_wiki:
            match += constant
            constant *= decay
    return match

def select_nearest(short_form: str, wikipages):
    id = 0
    lst = 0
    for idx, wiki in enumerate(wikipages):
        if matching(short_form, wiki) > lst:
            id = idx
            lst = matching(short_form, wiki)
    return wikipages[id]

def select_nearsest_shortest_withspace(short_form:str, wikipages):
    mx = 0
    pos = []
    # print(short_form, wikipages)
    for wiki in wikipages:
        # print(matching_nospace(short_form, wiki))
        if matching_nospace(short_form, wiki) > mx:
            mx = matching_nospace(short_form, wiki) 
    # print(mx)
    for wiki in wikipages:
        # print(matching_nospace(short_form, wiki))
        if matching_nospace(short_form, wiki) == mx:
            pos.append(wiki)
    # print(pos)
    return min(pos, key=lambda x: len(x))


def date_transform(text, question):
    text = text.lower().translate(str.maketrans('','',string.punctuation))
    words = text.split()
    lookup = {'năm': '', 'tháng': '', 'ngày': '', 'mùng': ''}
    for idx, w in enumerate(words):
        if w in lookup and idx+1 < len(words):
            if(words[idx+1].isnumeric()):
                lookup[w] = words[idx+1]
    ans = ""
    lisw = ["ngày", "tháng", "năm"]
    lisq = []
    for w in lisw:
        if w in question:
            lisq.append(w)
    if len(lisq) == 3 or len(lisq) == 2:
        # day and month or full day -> Only take what it asked for
        for w in lisq:
            if lookup[w] != "":
                ans += w + " " + lookup[w] + " "
            elif w == 'ngày' and lookup['mùng'] != "":
                ans += 'mùng' + ' ' + lookup["mùng"] + " "
    elif len(lisq) == 1:
        if(lisq[0] == "năm"):
            if lookup[lisq[0]] != "":
                ans += lisq[0] + " " + lookup[lisq[0]]
                return ans
            else:
                # There is no năm inside the answer text, have to search for literal
                for d in words:
                    if d.isnumeric():
                        return "năm" + " " + d
        elif(lisq[0] == "tháng"):
            if lookup[lisq[0]] != "":
                ans += lisq[0] + " " + lookup[lisq[0]]
                return ans
            else :
                for d in words:
                    if d.isnumeric():
                        return "tháng" + " " + d
        # question on day -> full day
        for w in lisw:
            if lookup[w] != "":
                ans += w + " " + lookup[w] + " "
            elif w == 'ngày' and lookup['mùng'] != "":
                ans += 'mùng' + ' ' + lookup["mùng"] + " "
    else:
        # Asking a date but there is no indicator month, year, or day -> full day
        for w in lisw:
            if lookup[w] != "":
                ans += w + " " + lookup[w] + " "
            elif w == 'ngày' and lookup['mùng'] != "":
                ans += 'mùng' + ' ' + lookup["mùng"] + " "
    return ans.strip()

def checktype(text, question):
    text = text.lower().translate(str.maketrans('','',string.punctuation))
    words = text.split()
    """Check if the text is a date or a number."""
    time_indicators = ["năm nào", "năm mấy", "năm bao nhiêu", "ngày bao nhiêu", "ngày tháng năm nào", "thời điểm nào", "ngày tháng âm lịch nào hằng năm", "thời gian nào", "lúc nào", "giai đoạn nào trong năm"]
    if any(idc in question.lower() for idc in time_indicators):
        return 2
    if "có bao nhiêu" in question:
        for d in words:
            if d.isnumeric():
                return 1
    if text == "":
        return 3
    words = text.split()
    for w in words:
        if w == 'năm' or w == 'tháng' or w == 'ngày':
            return 2
    if len(words) == 1 and words[0].isdigit():
        return 1
    return 0

