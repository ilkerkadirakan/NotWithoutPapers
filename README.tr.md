# NotWithoutPapers

Papers Please temalı bir reinforcement learning (RL) ortamı.  
Amaç: Ajanın (sınır memuru), **kısmi gözlemlenebilirlik** altında doğru `APPROVE / DENY` kararları almayı öğrenmesi.

Bu repo’da ana eğitim algoritması **PPO (Stable-Baselines3)**.

## 1. Projenin Kısa Özeti

Bu projede her episode bir iş gününü temsil eder:

- Günlük kural seti üretilir (hangi ülkeler kabul, permit gerekli mi).
- Başvuru sahipleri sırayla gelir.
- Doküman alanları başlangıçta gizlidir.
- Ajan isterse inspect aksiyonlarıyla bilgi açar.
- Sonra `APPROVE` veya `DENY` verir.
- Doğru karar ödül getirir, yanlış karar ceza getirir, inspect zaman ve küçük maliyet tüketir.

Bu, PPO için klasik bir trade-off problemidir:

- Daha çok inspect -> daha fazla bilgi, ama zaman/maliyet.
- Az inspect -> hızlı karar, ama hata riski.

## 2. Neden Bu Mimari?

Mimari katmanlı kuruldu:

- `env/`: ortam ve domain mantığı
- `train/`: PPO eğitimi ve callback
- `eval/`: metrik hesaplama ve evaluation loop
- `tests/`: sözleşme ve determinism testleri
- `scripts/`: yardımcı scriptler
- `main.py`: platform bağımsız tek giriş noktası

Bu ayrım şu faydayı sağlar:

- Environment değişikliği ile eğitim kodu birbirine karışmaz.
- Testler kolay yazılır ve sözleşme kırılmaları erken yakalanır.
- Tekrarlanabilir deney akışı kurulur.

## 3. Hızlı Başlangıç

## 3.1 Kurulum

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS:

```bash
python -m venv .venv
./.venv/bin/python -m pip install -U pip
./.venv/bin/python -m pip install -r requirements.txt
```

## 3.2 Smoke test

Platform bağımsız:

```bash
python main.py smoke --skip-pytest
```

## 3.3 Testler

```bash
python -m pytest
```

## 3.4 Kısa eğitim

```bash
python main.py train --total-timesteps 20000 --n-envs 4 --eval-episodes 20
```

## 3.5 Model değerlendirme

```bash
python main.py eval --model-path artifacts/ppo_papers_please.zip --episodes 50
```

## 4. Platform Bağımsızlık Notu

Repodaki ana kullanım yolu:

- `python main.py ...`

`scripts/*.ps1` dosyaları sadece Windows için kolaylık scriptleridir.  
Zorunlu değiller; istersen kullan, istersen tamamen `main.py` ile devam et.

## 5. Dosya Dosya Ne Var?

## 5.1 Root

- `main.py`: `train / eval / smoke` alt komutlarını yöneten ana CLI.
- `requirements.txt`: bağımlılıklar.
- `pytest.ini`: test keşif ayarları.
- `AGENTS.md`: proje sözleşmesi ve mühendislik kuralları.
- `ARCHITECTURE.md`: üst seviye mimari dokümanı.

## 5.2 `env/`

- `constants.py`:
  - Country listesi, field listesi, action ID sabitleri.
  - **Action ID sırasını değiştirmemek kritik.**
- `domain.py`:
  - `Rules`, `Applicant` dataclass’ları.
  - `oracle_is_legal(...)`: ground-truth legalite kontrolü.
- `sampling.py`:
  - Günlük rule örnekleme.
  - Applicant üretimi.
  - DENY oranını belirli bandda tutma (stabil öğrenme sinyali için).
- `papers_env.py`:
  - Gymnasium ortamının tamamı (`reset`, `step`, observation builder).

## 5.3 `train/`

- `train_ppo.py`:
  - PPO modelini kurar, eğitir, kaydeder, sonra eval yapar.
- `callbacks.py`:
  - Episode bazlı metrikleri `info["episode_stats"]` içinden toplayıp loglar.

## 5.4 `eval/`

- `evaluate.py`:
  - Deterministic policy ile episode loop’u döner.
- `metrics.py`:
  - accuracy, FAR, FRR, inspect frequency, mean reward hesapları.

## 5.5 `tests/`

- `test_env_contract.py`: observation/action sözleşmesi.
- `test_episode_flags.py`: `terminated/truncated` ve `episode_stats`.
- `test_determinism.py`: aynı seed ile reset determinism.

## 5.6 `scripts/`

- `smoke_test.py`: minimal runtime check.
- `run_smoke.ps1`, `run_train.ps1`, `run_eval.ps1`: Windows helper scriptleri.

## 6. Environment Sözleşmesi (Kritik)

Action ID’leri:

- `0`: APPROVE
- `1`: DENY
- `2`: INSPECT_COUNTRY_ALLOWED
- `3`: INSPECT_HAS_PERMIT
- `4`: INSPECT_EXPIRY_VALID
- `5`: INSPECT_NAME_MATCH

Observation vektörü:

- günlük kural durumu
- mevcut başvuru sahibi ülkesi (one-hot)
- reveal durumu (unknown/true/false)
- normalize `time_left`
- normalize kalan başvuru sayısı

Not:

- Bu sözleşme değişirse, train/eval kodu ve testler aynı committe güncellenmeli.

## 7. Episode Akışı (Adım Adım)

`reset()`:

1. Seed uygulanır.
2. Günlük rules örneklenir.
3. Günlük queue üretilir.
4. Zaman budget ve reveal/state sıfırlanır.
5. İlk observation döner.

`step(action)`:

1. Aksiyon karar mı inspect mi ayrılır.
2. Inspect ise ilgili field reveal edilir, inspect cezası ve zaman düşer.
3. Decision ise oracle ile doğruluk kontrol edilir, reward/ceza yazılır.
4. Episode sonu:
   - queue biterse `terminated=True`
   - zaman biterse `truncated=True`
5. Bitişte `info["episode_stats"]` set edilir.

## 8. Reward Tasarımı

Ortamda varsayılan olarak:

- doğru karar: `+4.0`
- false accept: `-15.0`
- false reject: `-8.0`
- inspect maliyeti: `-0.1`

Bu dağılım false accept’i daha ağır cezalandırır.  
Yani ajanın “kaçırma” hatasına karşı daha konservatif olmasını teşvik eder.

## 9. PPO Bu Projede Nasıl Kullanılıyor?

`train/train_ppo.py` içinde:

- Policy: `MlpPolicy`
- Vectorized env: `make_vec_env`
- Örnek hiperparametreler:
  - `n_steps=256`
  - `batch_size=256`
  - `gamma=0.99`
  - `learning_rate=3e-4`
  - `ent_coef=0.01`

Eğitim sonunda:

1. model `artifacts/*.zip` olarak kaydedilir,
2. deterministic eval çalıştırılır,
3. metrik özeti yazdırılır.

## 10. Metrikleri Nasıl Yorumlamalısın?

- `decision_accuracy`: genel doğru karar oranı.
- `false_accept_rate`:
  - güvenlik açısından kritik metrik,
  - yüksekse illegal başvurular geçiyor.
- `false_reject_rate`:
  - gereksiz ret oranı,
  - yüksekse legal başvurular zarar görüyor.
- `inspection_frequency`:
  - karar başına ortalama inspect sayısı.
  - Çok yüksekse ajan “aşırı inceleme” yapıyor olabilir.
- `episode reward`:
  - toplam hedef metrik.
  - Ama tek başına bakmak bazen yanıltıcı olabilir; alt metriklerle birlikte okunmalı.

## 11. Reproducibility (Tekrar Üretilebilirlik)

Bu projede tekrar üretilebilirlik için:

- seed kullanılıyor (`env` + `train` + `eval`).
- testlerde determinism kontrolü var.

Yine de farklı cihaz/BLAS/PyTorch backend farkları küçük sapmalar üretebilir.  
Bu normaldir; trend ve metrik davranışı esas alınmalıdır.

## 12. Sık Karşılaşılan Hatalar

`pytest` bulunamadı:

```bash
python -m pip install -r requirements.txt
```

Model yolu `.zip.zip` hatası:

- CLI ve scriptler bu duruma göre normalize edildi.
- Yine de path’i açık ve doğru vermek en güvenlisi.

Progress bar import hatası (`rich/tqdm`):

- `train` varsayılanında progress bar kapalı.
- Açmak için `--progress-bar` kullan.

## 13. Geliştirme Sırası (Öneri)

Projeyi anlamak için bu sırayla oku:

1. `main.py`
2. `train/train_ppo.py`
3. `env/papers_env.py`
4. `env/sampling.py`
5. `eval/metrics.py`
6. `tests/`

## 14. Sonraki Adımlar

Önerilen iyileştirmeler:

1. Root `README` içine kısa bir görsel akış diyagramı (Mermaid).
2. `configs/` klasörü ekleyip tüm parametreleri yaml’dan okumak.
3. GitHub Actions ile `smoke + pytest` otomasyonu.
4. Baseline policy karşılaştırmaları (random / heuristic / PPO).
