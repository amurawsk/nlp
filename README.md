# NLP projekt - Adrian Murawski, Przemysław Walecki

## Temat
Porównanie różnych podejść kodowania etykiet z wykorzystaniem modelu (RoBERTa) do określania podobieństwa semantycznego (iSTS - https://alt.qcri.org/semeval2016/task2/) dwóch zdań w języku angielskim. Odniesienie się do wyników z obronionej pracy magisterskiej/projektu.

Zadania:
1. Zapoznać się z pracą magisterską/projektem i przygotowanym rozwiązaniem.
2. Zapoznać się z wykorzystywanym modelem RoBERTa
3. Zmodyfikować kod bazowy wykorzystujący model RoBERTa zgodnie z informacjami przekazanymi przez prowadzącego (udostępniona zostanie szczegółowa informacja na temat wymaganych modyfikacji - porównanie różnych sposobów kodowania etykiet: łączenie etykiet, oddzielne etykiety)
4. Policzyć miary F dla stworzonego systemu dla wszystkich dostępnych danych z SemEval 2015 Interpretable STS z pojedynczymi relacjami oraz SemEval 2016 - Interpretable STS z rozbudowanymi relacjami.

## Definicja problemu
STS (Semantic Textual Similarity) to zadanie, w którym system ocenia, jak bardzo dwa zdania są podobne znaczeniowo, na skali 0–5. Taki wynik to jedna liczba i nic nie wyjaśnia. iSTS rozszerza ocenę podobieństwa dwóch zdań o wyjaśnienie: zdania są dzielone na fragmenty, fragmenty są dopasowywane, a każda para dopasowanych fragmentów dostaje typ relacji i ocenę podobieństwa. W projekcie podział i dopasowania są dane, więc model RoBERTa-large [3] (enkoder Transformer, ok. 355 mln parametrów) dostaje parę fragmentów i przewiduje dwie etykiety:
- typ relacji (type): EQUI, OPPO, SPE1, SPE2, SIMI, REL lub NOALI,
- ocenę podobieństwa (score): liczbę całkowitą od 0 do 5.

Etykiety są od siebie zależne: NOALI ma zawsze ocenę 0, a EQUI ocenę 5, więc dopuszczalnych par jest 27. Problemem jest wybór sposobu zakodowania tych dwóch etykiet na wyjściu modelu:
- etykiety oddzielne – dwie głowice, osobno dla oceny i typu,
- etykiety łączone – kodowanie 2-hot z jednym softmaxem albo jedna klasa ocena–typ (42 lub 27 klas).

Celem jest sprawdzenie, jak sposób kodowania wpływa na miary F1 przy niezmienionych pozostałych ustawieniach, oraz porównanie wyników z pracą Kopyta [4] i projektem Chmurzyńskiego [5].

## Wiedza domenowa
Zadanie iSTS pojawiło się wstępnie w SemEval 2015, na zbiorach Headlines i Images i tylko z dopasowaniami 1:1 [2]. W SemEval 2016 stało się osobnym zadaniem: zniesiono ograniczenie 1:1, zaktualizowano wytyczne i dodano zbiór Answers-Students [1].

| Typ | Znaczenie | Ocena |
| :--- | :--- | :---: |
| **EQUI** | fragmenty równoważne znaczeniowo | 5 |
| **OPPO** | znaczenia przeciwstawne | 1–5 |
| **SPE1 / SPE2** | podobne, fragment ze zdania 1 / 2 bardziej szczegółowy | 1–5 |
| **SIMI** | podobne, ale bez powyższych relacji | 1–5 |
| **REL** | powiązane, ale niepodobne | 1–5 |
| **NOALI** | fragment bez odpowiednika | 0 |

Ocena 5 oznacza znaczenie równoważne, a 0 brak związku.

Te dwie etykiety nie są niezależne. NOALI ma zawsze ocenę 0, a EQUI zawsze ocenę 5. Pozostałe typy mają oceny 1–5. Ze wszystkich 42 kombinacji (7 typów × 6 ocen) dopuszczalnych jest tylko 27. Problemem jest kodowanie tych dwóch etykiet na wyjściu sieci, żeby wyniki były najlepsze. Możliwości są cztery:
- 2-hot, jeden softmax (Chmurzyński): jedna warstwa wyjściowa z 14 liczbami (6 dla oceny + 8 dla typu) i wspólny softmax. Ocena i typ „konkurują” w jednym rozkładzie prawdopodobieństwa.
- dwie osobne głowice (Kopyt ii): jedna klasyfikuje ocenę (6 klas), druga typ (7 klas). Każda ma własny softmax, a straty się sumuje. Model nie wie wprost, że oceny i typy się łączą, więc może przewidzieć np. EQUI z oceną 2.
- jedna etykieta łączona: każda para ocena–typ to osobna klasa. W wersji 42 klas sieć może nadal wybrać parę niedopuszczalną. W wersji 27 klas niedopuszczalnych par po prostu nie ma.
- dekodowanie z ograniczeniami (constraint decoding): model z dwiema głowicami trenowany jest bez zmian. Dopiero przy odczycie wyniku wybierana jest najlepsza para spośród 27 dopuszczalnych, zamiast brać osobno najlepszy typ i najlepszą ocenę.

Wszystkie te wersje mają porównywalną architekturę i ten sam trening. Różnią się tylko kodowaniem wyjścia, i ten wpływ zamierzamy zbadać.

## Opis rozwiązania
Rozwiązanie zostanie podzielone na dwa etapy. Z dwóch wariantów rozwiązania Kopyta realizujemy wariant ii), czyli klasyfikację zarówno oceny (score), jak i typu relacji (type).

### Część A
Baza: rozwiązanie joint (Chmurzyński, najlepsza wersja).
- Krok 1. Uruchomienie rozwiązania joint (Chmurzyński, najlepsza wersja).
- Krok 2. Przerobienie rozwiązania z kroku 1 tak, aby głowice były zgodne z Kopyt ii): głowica klasyfikująca dla score (Negative log-likelihood, NLL, loss) i głowica klasyfikująca dla 7 typów relacji (NLL loss), trenowane razem (jointly) jako Loss NLL + Loss NLL.
- Krok 3. Przerobienie rozwiązania z kroku 1 tak, aby była jedna głowica one-hot:
  - 3.1 – z 42 elementami pokrywającymi wszystkie kombinacje score i type,
  - 3.2 – z 27 elementami pokrywającymi tylko dopuszczalne kombinacje score i type (np. EQUI może mieć tylko ocenę 5, czyli EQUI-5; niedopuszczalne jest np. EQUI-2).
- Krok 4. Constraint decoding dla rozwiązania z kroku 2, ograniczający generowane wyniki do 27 dopuszczalnych kombinacji.

Zmieniamy tylko głowice. Reszta (w szczególności hiperparametry i schedule) zostaje jak u Chmurzyńskiego, aby w jednym kroku zmieniał się tylko jeden czynnik.

### Część B
Baza: Kopyt, z jego hiperparametrami i schedule, ale bez dotrenowania RoBERTa-large na STS-B; reszta bez zmian.
- Krok 5. Przeliczenie podejścia Kopyt ii) (klasyfikacja score + klasyfikacja 7 typów, Loss NLL + Loss NLL) bez dotrenowania RoBERTa-large na STS-B.
- Krok 6. Przerobienie kroku 5 na joint jak w pracy Chmurzyńskiego: kodowanie 2-hot z jednym softmaxem, 14 logitów (6 score + 8 type).

### Ewaluacja
Każdy eksperyment uruchomimy 5 razy z różnymi seedami. Miary F1 (F1_A, F1_AT, F1_AS, F1_AST i F1_AST-NP) policzymy oficjalnym skryptem SemEval na danych testowych 2016 i 2015, dla dopasowań złotych (gs), a dla 2016 także dla dopasowań z sieci Maji [6]. Wyniki podamy jako średnią i odchylenie standardowe z 5 uruchomień, a istotność różnic między eksperymentami sprawdzimy testami statystycznymi (test t Welcha, test Manna–Whitneya) [7]. Nowe wyniki dopiszemy do tabeli obok dotychczasowych rozwiązań.

## Implementacja
Implementacja zostanie zrealizowana w języku Python.
- Część A: notatnik Chmurzyńskiego przeniesiony do skryptów; PyTorch i Hugging Face transformers (roberta-large), wymienna głowica i funkcja straty dla każdego kodowania.
- Część B: aplikacja Kopyta (fairseq, roberta.large) bez kroku STS-B, z dodaną głowicą 2-hot.
- Wspólna ocena: zapis plików .wa, uruchomienie skryptów Perl (evalF1), zebranie wyników F1 do CSV i testy istotności (SciPy).

(pozostałe elementy implementacji zostaną zawarte w dokumentacji końcowej)

## Instrukcja obsługi
(dokumentacja końcowa)

## Testy
(dokumentacja końcowa)

## Wnioski
(dokumentacja końcowa)

## Literatura
1. E. Agirre i in., „SemEval-2016 Task 2: Interpretable Semantic Textual Similarity”, Proceedings of SemEval-2016, 2016. aclanthology.org/S16-1082
2. E. Agirre i in., „SemEval-2015 Task 2: Semantic Textual Similarity, English, Spanish and Pilot on Interpretability”, Proceedings of SemEval 2015, 2015. aclanthology.org/S15-2045
3. Y. Liu i in., „RoBERTa: A Robustly Optimized BERT Pretraining Approach”, arXiv:1907.11692, 2019. arxiv.org/abs/1907.11692
4. M. Kopyt, „Określanie interpretowalnego podobieństwa semantycznego dla języka angielskiego”, praca magisterska, Politechnika Warszawska, 2025.
5. M. Chmurzyński, H. Soroka, M. Tymoftyjewicz, dokumentacja końcowa projektu NLP i notatnik z kodem, Politechnika Warszawska, 2025.
6. S. Maji i in., „Logic Constrained Pointer Networks for Interpretable Textual Similarity”, IJCAI-20, 2020. doi.org/10.24963/ijcai.2020/333
7. R. Dror i in., „The Hitchhiker’s Guide to Testing Statistical Significance in Natural Language Processing”, ACL 2018. aclanthology.org/P18-1128
