# Training Strategy (PPO)

Bu dokuman, ortam karmasikligini kaybetmeden egitimi stabil tutmak icin izlenecek resmi protokoldur.

## Hedef

Temel hedefi tek satirda tanimla:

- Zorunlu: `decision_coverage >= 0.90`
- Kalite: `decision_accuracy` maksimize
- Risk kontrolu: `false_accept_rate` dusuk kalmali

Pratik gecis hedefleri:

- Stage gecisi icin: `coverage >= 0.90` ve `accuracy >= 0.75`

## Optimization Kurali (Lexicographic)

Model secimi su sirayla yapilir:

1. Coverage kuralini saglayan runlar disindakiler elenir.
2. Kalanlar icinde `decision_accuracy` yuksek olan secilir.
3. Esitlikte `false_accept_rate` dusuk olani secilir.
4. Hala esitlik varsa `inspection_frequency` dusuk olani secilir.

Not: `episode_reward` tek basina model secim metriði degildir.

## Stage Plan

### Stage A (Stabil On-Egitim)

Amac: ajanin karar verme davranisini stabil ogrenmesi.

- `mid_day_update_prob = 0.0`
- `inspect_error_prob = 0.0`
- `inspect_miss_prob = 0.0`
- `max_inspects_per_applicant = 2`
- Coverage hard constraint aktif (`coverage_hard_threshold = 0.9`)

Gecis kosulu:

- Son 3 evalin ortalamasi ile `coverage >= 0.90`
- Son 3 evalin ortalamasi ile `accuracy >= 0.75`

### Stage B (Gercekci Fine-Tune)

Amac: Stage A'da ogrenilen policy'yi daha zor dagilimde saglamlastirmak.

- `mid_day_update_prob > 0` (or. `0.3 -> 0.6`)
- `inspect_error_prob > 0` (or. `0.05`)
- `inspect_miss_prob > 0` (or. `0.05`)
- Diger parametreler sabit tutulur.

Gecis kosulu:

- `coverage >= 0.90` korunmali
- `accuracy` Stage A'ya gore kabul edilebilir sinirda kalmali

## Degisken Kontrolu (Tek Degisken Kurali)

Ayni anda birden fazla reward parametresi degistirme.

Oncelikli ayar sirasi:

1. `p_false_accept`
2. `r_correct`
3. `p_false_reject`

Her denemede sadece 1 parametre degis, digerlerini sabit tut.

## Ablation Sirasi

Ozellikleri ac/kapa test ederken su sirayi kullan:

1. `max_inspects_per_applicant`
2. `coverage_hard_threshold + coverage_hard_penalty`
3. `mid_day_update_prob`
4. `inspect_error_prob / inspect_miss_prob`

Her adimda su farklari raporla:

- `decision_coverage`
- `decision_accuracy`
- `false_accept_rate`
- `false_reject_rate`
- `inspection_frequency`

## Run Protokolu

Her konfig icin en az:

- 3 farkli seed
- kisa run (hizli eleme)
- orta run (aday secimi)

Ornek:

1. Kisa: `20k` timesteps x 3 seed
2. Orta: `100k` timesteps x 3 seed
3. En iyi aday: `200k+` timesteps

## Model Secim Skoru (Opsiyonel Yardimci)

Coverage gate gectikten sonra yardimci skor:

`score = decision_accuracy - 0.5 * false_accept_rate`

Kural:

- Eger `coverage < 0.90`, run otomatik basarisiz.

## Dur/Kontrol Kriterleri

Asagidaki durumlarda complexity artirma, once dengele:

- Coverage tamam ama accuracy tek aksiyona cokuyor (hep approve/deny)
- Inspection frequency asiri uc degerlere gidiyor
- 3 seedde tutarsiz sonuclar var

## Kayit Formati (Her Run)

Asagidaki alanlari bir tabloya yaz:

- tarih
- stage
- seed
- timesteps
- env parametreleri
- reward parametreleri
- coverage
- accuracy
- false_accept
- false_reject
- inspection_frequency
- notlar

## Uygulama Notu

Su an CLI, tum env parametrelerini disaridan almiyor olabilir.
Bu durumda asamali stratejiyi uygulamak icin iki yol var:

1. Gecici olarak `train/train_ppo.py` icindeki `env_kwargs` degerlerini stage'e gore guncellemek
2. Sonraki adimda `main.py train` ve `train/train_ppo.py` icin env parametreleri CLI argumanlari eklemek

Tercih: Uzun vadede yol 2.
