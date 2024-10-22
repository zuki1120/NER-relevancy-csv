* test.py 更改為當搜尋詞沒有對應實體時ner_relevancy_1及ner_relevancy_2都會標成沒有實體、若單一商品名稱沒有對應實體則在它那邊的ner_relevancy標成沒有實體

* test_brute-force.py 搜尋詞(沒有經過NER)與商品名稱的'品牌'、'產品'任一實體完全對應則標2

* check_NER.csv 檢查沒有實體的版本

* brute.csv 暴力解版本

* modify.csv 只加了權重

* relevancy.csv 助教版本
