import string
from underthesea import ner

def dry(item):
  dates_idc = ["năm", "tháng", "mùng", "ngày"]
  res_ans = []
  for ans in item['answers']:
    tmp = ans.translate(str.maketrans('','', string.punctuation))
    words = tmp.split()
    is_spec = False
    for w in words:
      if w in dates_idc or w.isnumeric():
        is_spec = True
        break
    res = list()
    # print(is_spec)
    if is_spec:
      for w in words:
        if w in dates_idc or w.isnumeric():
          res.append(w)
    else:
      tmp_ner = ner(tmp, deep=True)
      # print(tmp_ner)
      for w in tmp_ner:
        if w['entity'] != 'O':
          res.append(w['word'])
    ans = " ".join(res)
    res_ans.append(ans)
  item['answers'] = res_ans
  return item

def main():
  data = {
            "id": "testa_560",
            "question": "Trần Khâm là tên gọi khác của vị vua nào?",
            "answer": [
                "Djoser",
                "pharaon Amenemhat VI",
                "Ai Cập",
                "vua Bảo Đại",
                "\"Neferkare",
                "Ramesses",
                "vua Djoser",
                "vua Djoser",
                "Trần Thái Tông",
                "Nhân Tông"
            ],
            "scores": [
                1.7692718401463026e-08,
                3.895761047090218e-09,
                9.054956784382284e-10,
                0.000908917048946023,
                2.0464647771528677e-12,
                3.9916199234824035e-09,
                1.1994133464199308e-09,
                1.1994133464199308e-09,
                3.050642044399865e-05,
                0.012792181223630905
            ],
            "passage_scores": [
                0.15817099571228027,
                0.1508520030975342,
                0.13289400100708007,
                0.13287899971008302,
                0.13032400131225585,
                0.13029399871826172,
                0.1301039981842041,
                0.13010398864746095,
                0.12894800186157226,
                0.12523799896240234
            ],
            "candidate_retrieving_wikipages": [
                [
                    "wiki/Djoser"
                ],
                [
                    "wiki/Seankhibtawy_Seankhibra"
                ],
                [
                    "wiki/Khufu"
                ],
                [
                    "wiki/Nguyễn_Văn_Sâm"
                ],
                [
                    "wiki/Neferkare_VII"
                ],
                [
                    "wiki/Yakareb"
                ],
                [
                    "wiki/Seth-Peribsen"
                ],
                [
                    "wiki/Sekhemib-Perenmaat"
                ],
                [
                    "wiki/Bang_giao_Đại_Việt_thời_Trần"
                ],
                [
                    "wiki/Trần_Nhân_Tông"
                ]
            ]
        }
  # print(dry(data))
  print(dry(data))

if __name__ == "__main__":
  main()