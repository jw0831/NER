# 개체명 인식(NER) - 일반적인 개체명인식 모델 개발기

## 개발 순서
### 1. 개체명 인식 데이터셋 조사
- 엑소브레인
- klue ner
- klp2016
- 국립국어원
- 추가 : 119긴급신고접수용으로 직접 고안한 데이터셋

#### 데이터 분석 및 일반적 태그 정의 과정
![](<./image_file/general_ner_define.png>)

#### 최종 태그 정립 : 대분류(main category)로 LOC, ORG, QTY에 개체명들이 새로 정의한 (LMK, BRD, MET)등으로 분류할 수 있을 정도로 섞여 있으므로 대분류에서 대분류로 한번 더 나누어줌

|개체명	|태그	|정의|
|---|---|---|
사람	   | PER	|사람, 미술가, 경찰, 사람이름, 
사건	   | EVT	|예비군훈련, 교통사고, 전쟁, 축제, IMF, 운동, 연주회, 세미나, 
날짜	   | DAT	|년월일, 로마시대, 1945년, 지난해 하반기, 십칠 세기 이후, 20년 전, 10년 후
시간	   | TIM	|시간, 기간, 몇 분, 시간 동안, 육분 내지 칠분 정도, 
기관/단체	|ORG	|병원, 학교, 시설, 어린이집, 학원, 법원, 
장소	   | LOC	|주소, 몇동 몇호, 종로3가, 수표동, 도로명, 읍, 면, 동, 
랜드마크	|LMK	|건물명, 광화문, 보신각, 빌딩명, 시그니쳐타워, 한화빌딩, 시그니엘서울, 롯데월드타워, 청계천 배를린광장, 롯데시티호텔명동, 뱅뱅사거리, 은행사거리, 
브랜드	    |BRD	|구찌, 삼성전자, 삼성조명, 스타벅스, 대련집, 서브웨이, 순심김밥, 식당명, 옷 브랜드, 전자제품 브랜드, 자동차 브랜드, 
질병	    |DIS	|병명, 코로나, 감기, 아픈증상(열, 콧물, 기침, 가래), 맹장수술, 구내염,
수량	    |QTY	|몸무게, 수량, 몇 명, 몇 번, 
비율	    |PNT	|약 43배, 약 팔십 퍼센트 정도, 0.003%, 
측량	    |MET	|400m, 육 키로미터, 36만㎡, 오 미터 짜리, 90㎜Hg
통화	    |MNY	|49억700만 달러, 1만8000~1만8800원


### 2. 각 데이터셋 마다 태그된 방식이 상이함으로 한가지 방식으로 통일 후 한번에 사용할 개체와 사용하지 않을 개체를 필터링 및 매핑

- 엑소브레인
    - 변환 전
        ![](<./image_file/exo_data_viz.png>)
    - 변환 후 
        ![](<./image_file/exo_data_viz_after.png>)

- klue ner
    - 변환 전
        ![](<./image_file/klue_data_viz.png>)
    - 변환 후 
        ![](<./image_file/klue_data_viz_after.png>)

- klp2016
    - 변환 전
        ![](<./image_file/klp_data_viz.png>)
    - 변환 후 
        ![](<./image_file/klp_data_viz_after.png>)

- 국립국어원
    - 변환 전
        ![](<./image_file/kuklip_data_.png>)
        ![](<./image_file/kuklip_data_viz.png>)
    - 변환 후 
        ![](<./image_file/kuklip_data_viz_after.png>)


#### 개체명 태그를 통일 후 분석한 결과 (119 미포함, 0개는 새로 정의한 태그)
- 현재 갖고 있는 데이터셋 개수 : 196589 태깅된 문장 <br>

|개체명|태그|개수|
|-----|----|----|
|사람           |PER| 21086|
|사건           |EVT|  2391|
|날짜           |DAT| 10484|
|기관/단체      |ORG| 13431|
|장소           |LOC|  5829|
|랜드마크/건물명 |LMK|     0|
|시간           |TIM|  2062|
|질병           |DIS|     0|
|브랜드         |BRD|     0|
|수량           |QTY| 18272|
|측량           |MET|     0|
|비율           |PNT|     0|
|통화           |MNY|     0|

### 3. 데이터 라벨링을 수정하기 위한 작업
- [Label Studio](https://labelstud.io/)를 통해 새로 정립한 개체로 수정하는 작업 후 학습을 진행할 것
- 예시
    ![](<./image_file/labe_studio_ex.png>)

#### 라벨스튜디오로 import 하기 위한 통일된 태그들을 일괄적으로 json 파일로 변환하는 과정이 필요
1. 라벨스튜디오로 아무 텍스트나 업로드 후 json으로 다운로드 및 json 형식 분석
```
# 라벨스튜디오에서 내보낸 json파일 형식
# label studio data *.json
[{
    "id":20018,
    "annotations":[
                  {
                    "id":2193,
                    "completed_by":1,
                    "result": [
                                {
                                "value":{"start":0,"end":6,"text":"롯데하이마트","labels":["ORG"]},
                                "id":"8u6eekdVhy", # 이 id는 달라야 업로드 할때 여러 태그를 인식함
                                "from_name":"label",
                                "to_name":"text",
                                "type":"labels",
                                "origin":"manual"
                                },
                                {
                                "value":{"start":10,"end":13,"text":"이동우","labels":["PER"]},
                                "id":"8xNcT4cKCy",
                                "from_name":"label",
                                "to_name":"text",
                                "type":"labels",
                                "origin":"manual"
                                }
                              ],
                    "was_cancelled":false,
                    "ground_truth":false,
                    "created_at":"2023-01-05T02:21:37.962532Z",
                    "updated_at":"2023-01-05T02:21:54.384899Z",
                    "lead_time":14.261,
                    "prediction":{},
                    "result_count":0,
                    "task":20018,
                    "parent_prediction":null,
                    "parent_annotation":null
                  }
                  ],
    "file_upload":"539369e9-new_18.txt",
    "drafts":[],
    "predictions":[],
    "data":{"text":"롯데하이마트(대표 이동우)가 18일부터 전국 460여 개 매장에 ‘보이는 ARS’ 결제 서비스를 도입한다. ‘보이는 ARS’ 결제 서비스는 기존 음성으로만 가능했던 ARS 카드 결제 방식을 보완해 고객이 스마트폰으로 직접 화면을 보며 카드 결제가 가능하도록 설계한 서비스다. ‘보이는 ARS’ 결제 서비스는 국내 가전 유통업계 가운데 롯데하이마트에서 최초 도입했다. "},
    "meta":{},
    "created_at":"2023-01-05T02:21:30.292467Z",
    "updated_at":"2023-01-05T02:21:54.400487Z","inner_id":1,"project":82,"updated_by":1
}]
```

2. json 형태 분석 후 기존 라벨링된 데이터를 json형태로 변환하는 작업 진행


```
def convert_taggedsent2json(df, debug=False):
    whole_json_converted = []
    ne_sent_counter = 0
    
    for idx, sent_tagged in enumerate(tqdm(df['tagged_sents'].iloc[:])):
        id = idx
        # print(id, sent_tagged)
        soup = BeautifulSoup(sent_tagged,'html.parser')
        links = soup.find_all("ne")
        if debug:
            print(sent_tagged)
            print(links)
            
            
        if len(links) > 0: # 사용하지 않을 개체명제거후 개체가 남는 문장인 경우만 진행
            ne_sent_counter+=1
            
            _result = []
            for link_idx, link in enumerate(links):
                
                _tag_dict = {}
                ne = link.attrs['tag']
                text = link.string
                
                re_link = str(link).replace('<ne','<NE').replace('</ne','</NE').replace('\"','\'').replace('&amp;','&').replace('&lt;','<').replace('&gt;','>').replace('*','\*').replace('(','\(').replace(')','\)')
                
                
                if re.search(re_link, sent_tagged):
                    matched = re.search(re_link, sent_tagged)
                    start = matched.span()[0]
                    end = start + len(text)
                    # sent_tagged = re.sub(re_link, text ,sent_tagged) # 이건 사용금지임 -> 겹치는 단어 있는경우 span오류 발생, 태그 겹쳐버림
                    sent_tagged = sent_tagged[:start] + text + sent_tagged[matched.span()[1]:]
                    
                    if debug:
                        print('######################################')
                        print(matched) 
                        print(matched.span()) 
                        print(sent_tagged)
                        print('######################################')
                
                _tag_dict['start'] = start
                _tag_dict['end'] = end
                _tag_dict['text'] = text
                _tag_dict['labels'] = [ne]
                    
                _outer_tag_dict = {
                                    "value": _tag_dict,
                                    "id":str(link_idx), # 이게 겹치면 다른 태그가 라벨스튜디오에 하나만보임
                                    "from_name":"label",
                                    "to_name":"text",
                                    "type":"labels",
                                    "origin":"manual"
                                  }
                _result.append(_outer_tag_dict)
                
                if debug:
                    print(re_link)
                    print(_result)
            
            _annotations = {"id":id,
                            "completed_by":1,
                            "result": _result,
                            "was_cancelled":False,
                            "ground_truth":False,
                            "created_at":"2023-01-05T02:21:37.962532Z",
                            "updated_at":"2023-01-05T02:21:54.384899Z",
                            "lead_time":14.261,
                            "prediction":{},
                            "result_count":0,
                            "task":55555,
                            "parent_prediction":'null',
                            "parent_annotation":'null'}
                            
            _datum = {"id":55555,
                    "annotations": [_annotations],
                    "file_upload":"539369e9-new_18.txt",
                    "drafts":[],
                    "predictions":[],
                    "data":{"text":sent_tagged},
                    "meta":{},
                    "created_at":"2023-01-05T02:21:30.292467Z",
                    "updated_at":"2023-01-05T02:21:54.400487Z","inner_id":id,"project":500,"updated_by":1}
                    

            
            # print(sent_tagged)
            # print(_datum)
            whole_json_converted.append(_datum)
    print('number of sents converted:',ne_sent_counter)
    return whole_json_converted

whole_json_converted_j1 = convert_taggedsent2json(df_j1)
whole_json_converted_j2 = convert_taggedsent2json(df_j2)
# whole_json_converted_j2 = convert_taggedsent2json(df, debug=True)
```

# json 파일로 저장 후 Label studio 에서 import로 업로드하면 됨 
- 한번에 import할 수 있는 용량이 200MB 초반이므로 나눠서 업로드 할것
```
with open('output_ner_gen_labelstudio_1.json','w', encoding='utf-8') as f:
    json.dump(whole_json_converted_j1, f, indent=2, ensure_ascii= False)
    
with open('output_ner_gen_labelstudio_2.json','w', encoding='utf-8') as f:
    json.dump(whole_json_converted_j2, f, indent=2, ensure_ascii= False)

```