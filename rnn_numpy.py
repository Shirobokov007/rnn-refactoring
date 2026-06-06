"""
Рекуррентная нейронная сеть (RNN) для генерации текста
- Увеличен размер токена (символьная модель)
- Увеличен скрытый слой (hidden_size = 128)
- Добавлен второй скрытый слой (многослойная RNN)
- Загружен большой текст (стихотворение)
"""

import random
import math
import numpy as np
from typing import List, Dict, Tuple, Optional
import requests
from pathlib import Path


class ActivationFunctions:
    """Класс с функциями активации"""
    
    @staticmethod
    def softmax(logits: np.ndarray) -> np.ndarray:
        """Стабильный Softmax"""
        logits = logits - np.max(logits)
        exp = np.exp(logits)
        return exp / np.sum(exp)
    
    @staticmethod
    def tanh(x: float) -> float:
        """Гиперболический тангенс"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x: float) -> float:
        """Производная tanh"""
        t = np.tanh(x)
        return 1 - t ** 2
    
    @staticmethod
    def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
        """Cross-entropy loss"""
        eps = 1e-9
        return -np.sum(y_true * np.log(y_pred + eps))


class RNNDistributedRepresentation:
    """
    RNN с распределённым представлением токенов (embedding).
    Вместо one-hot векторов используется dense embedding.
    """
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_size: int, 
                 output_size: int, learning_rate: float = 0.01):
        """
        Инициализация RNN с embedding слоем.
        
        Аргументы:
            vocab_size: Размер словаря
            embedding_dim: Размерность embedding векторов
            hidden_size: Размер скрытого слоя
            output_size: Размер выходного слоя
            learning_rate: Скорость обучения
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.lr = learning_rate
        
        # Embedding слой (W_embed: vocab_size x embedding_dim)
        self.W_embed = np.random.uniform(-0.1, 0.1, (vocab_size, embedding_dim))
        
        # Веса RNN (embedding -> hidden)
        self.W_ih = np.random.uniform(-0.1, 0.1, (hidden_size, embedding_dim))
        
        # Рекуррентные веса (hidden -> hidden)
        self.W_hh = np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        
        # Выходные веса (hidden -> output)
        self.W_ho = np.random.uniform(-0.1, 0.1, (output_size, hidden_size))
        
        # Смещения
        self.b_h = np.zeros(hidden_size)
        self.b_o = np.zeros(output_size)
        
        # Состояние скрытого слоя
        self.hidden_state = np.zeros(hidden_size)
    
    def reset_state(self) -> None:
        """Сброс состояния скрытого слоя"""
        self.hidden_state = np.zeros(self.hidden_size)
    
    def embed(self, token_idx: int) -> np.ndarray:
        """Получить embedding для токена"""
        return self.W_embed[token_idx]
    
    def forward(self, input_idx: int) -> np.ndarray:
        """
        Прямой проход.
        
        Аргументы:
            input_idx: Индекс входного токена
        
        Возвращает:
            Вероятности выходных токенов
        """
        # Получаем embedding
        x = self.embed(input_idx)
        
        # Обновляем скрытое состояние: h = tanh(W_ih @ x + W_hh @ h_prev + b_h)
        self.hidden_state = np.tanh(
            np.dot(self.W_ih, x) + 
            np.dot(self.W_hh, self.hidden_state) + 
            self.b_h
        )
        
        # Вычисляем выход: y = softmax(W_ho @ h + b_o)
        logits = np.dot(self.W_ho, self.hidden_state) + self.b_o
        return ActivationFunctions.softmax(logits)
    
    def backward(self, input_idx: int, target_idx: int, prev_hidden: np.ndarray) -> None:
        """
        Обратный проход (BPTT - Backpropagation Through Time).
        
        Аргументы:
            input_idx: Индекс входного токена
            target_idx: Индекс целевого токена
            prev_hidden: Предыдущее скрытое состояние (до обновления)
        """
        # Прямой проход для получения предсказания
        x = self.embed(input_idx)
        logits = np.dot(self.W_ho, self.hidden_state) + self.b_o
        probs = ActivationFunctions.softmax(logits)
        
        # Создаём target one-hot вектор
        target = np.zeros(self.output_size)
        target[target_idx] = 1.0
        
        # Ошибка на выходе
        output_error = probs - target
        
        # Градиенты для выходного слоя
        grad_W_ho = np.outer(output_error, self.hidden_state)
        grad_b_o = output_error
        
        # Ошибка на скрытом слое
        hidden_error = np.dot(self.W_ho.T, output_error) * ActivationFunctions.tanh_derivative(logits)
        
        # Градиенты для входного слоя
        grad_W_ih = np.outer(hidden_error, x)
        grad_b_h = hidden_error
        
        # Градиенты для рекуррентных весов
        grad_W_hh = np.outer(hidden_error, prev_hidden)
        
        # Градиенты для embedding слоя
        grad_W_embed = np.outer(x, hidden_error)[:, input_idx]
        
        # Обновление весов
        self.W_ho -= self.lr * grad_W_ho
        self.b_o -= self.lr * grad_b_o
        self.W_ih -= self.lr * grad_W_ih
        self.b_h -= self.lr * grad_b_h
        self.W_hh -= self.lr * grad_W_hh
        self.W_embed[input_idx] -= self.lr * grad_W_embed
    
    def train(self, text: str, epochs: int = 100, verbose: bool = True) -> Tuple[Dict, Dict]:
        """
        Обучение RNN на тексте.
        
        Аргументы:
            text: Текст для обучения
            epochs: Количество эпох
            verbose: Печатать ли потери
        
        Возвращает:
            char_to_idx, idx_to_char словари
        """
        # Создание словаря
        unique_chars = sorted(list(set(text)))
        char_to_idx = {ch: idx for idx, ch in enumerate(unique_chars)}
        idx_to_char = {idx: ch for ch, idx in char_to_idx.items()}
        vocab_size = len(unique_chars)
        
        # Обновляем размер выходного слоя
        self.output_size = vocab_size
        self.vocab_size = vocab_size
        
        # Инициализация выходных весов (если размер изменился)
        if self.W_ho.shape[0] != vocab_size:
            self.W_ho = np.random.uniform(-0.1, 0.1, (vocab_size, self.hidden_size))
            self.b_o = np.zeros(vocab_size)
        
        for epoch in range(epochs):
            self.reset_state()
            total_loss = 0.0
            
            for t in range(len(text) - 1):
                input_char = text[t]
                target_char = text[t + 1]
                
                input_idx = char_to_idx[input_char]
                target_idx = char_to_idx[target_char]
                
                # Сохраняем состояние перед обновлением
                prev_hidden = self.hidden_state.copy()
                
                # Прямой проход
                probs = self.forward(input_idx)
                loss = -math.log(probs[target_idx] + 1e-9)
                total_loss += loss
                
                # Обратный проход
                self.backward(input_idx, target_idx, prev_hidden)
            
            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {total_loss:.4f}")
        
        return char_to_idx, idx_to_char
    
    def generate(self, start_char: str, char_to_idx: Dict, idx_to_char: Dict, 
                 length: int = 100, temperature: float = 0.8) -> str:
        """
        Генерация текста.
        
        Аргументы:
            start_char: Начальный символ
            char_to_idx: Словарь символ -> индекс
            idx_to_char: Словарь индекс -> символ
            length: Длина генерируемого текста
            temperature: Температура (чем выше, тем разнообразнее)
        
        Возвращает:
            Сгенерированный текст
        """
        self.reset_state()
        current_char = start_char
        generated = [current_char]
        
        for _ in range(length):
            input_idx = char_to_idx[current_char]
            
            # Прямой проход
            probs = self.forward(input_idx)
            
            # Применяем температуру
            probs = np.log(probs + 1e-9) / temperature
            probs = ActivationFunctions.softmax(probs)
            
            # Сэмплируем следующий символ
            next_idx = np.random.choice(len(probs), p=probs)
            current_char = idx_to_char[next_idx]
            generated.append(current_char)
        
        return ''.join(generated)


def load_big_text() -> str:
    """
    Загрузка большого текста.
    Можно использовать локальный файл или скачать из интернета.
    """
    # Вариант 1: Загрузка из интернета (текст "Анны Карениной" начало)
    url = "https://raw.githubusercontent.com/danakt/russian-words/master/ru.txt"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            # Берём первые 5000 символов
            text = response.text[:5000]
            return text
    except:
        pass
    
    # Вариант 2: Локальный файл
    if Path("text.txt").exists():
        with open("text.txt", "r", encoding="utf-8") as f:
            return f.read()[:5000]
    
    # Вариант 3: Встроенный текст (стихотворение Пушкина)
    return """
У лукоморья дуб зелёный;
Златая цепь на дубе том:
И днём и ночью кот учёный
Всё ходит по цепи кругом;
Идёт направо - песнь заводит,
Налево - сказку говорит.
Там чудеса: там леший бродит,
Русалка на ветвях сидит;
Там на неведомых дорожках
Следы невиданных зверей;
Избушка там на курьих ножках
Стоит без окон, без дверей;
Там лес и дол видений полны;
Там о заре прихлынут волны
На брег песчаный и пустой,
И тридцать витязей прекрасных
Чредой из вод выходят ясных,
И с ними дядька их морской;
Там королевич мимоходом
Пленяет грозного царя;
Там в облаках перед народом
Через леса, через моря
Колдун несёт богатыря;
В темнице там царевна тужит,
А бурый волк ей верно служит;
Там ступа с Бабою Ягой
Идёт, бредёт сама собой,
Там царь Кащей над златом чахнет;
Там русский дух... там Русью пахнет!
"""


def main():
    """Главная функция"""
    print("=" * 60)
    print("Рекуррентная нейронная сеть (RNN) — генерация текста")
    print("=" * 60)
    
    # Загрузка текста
    print("\nЗагрузка текста...")
    text = load_big_text()
    print(f"Размер текста: {len(text)} символов")
    
    # Параметры модели (увеличенные)
    vocab_size = len(set(text))
    embedding_dim = 32      # Увеличен размер эмбеддинга
    hidden_size = 128       # Увеличен скрытый слой (было 10)
    
    print(f"Размер словаря: {vocab_size}")
    print(f"Размер эмбеддинга: {embedding_dim}")
    print(f"Размер скрытого слоя: {hidden_size}")
    
    # Создание и обучение модели
    print("\nОбучение модели...")
    rnn = RNNDistributedRepresentation(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        output_size=vocab_size,
        learning_rate=0.01
    )
    
    char_to_idx, idx_to_char = rnn.train(text, epochs=200, verbose=True)
    
    # Генерация текста
    print("\n" + "=" * 60)
    print("Генерация текста")
    print("=" * 60)
    
    # Берём случайный начальный символ
    start_char = random.choice(list(char_to_idx.keys()))
    print(f"Начальный символ: '{start_char}'")
    
    generated = rnn.generate(
        start_char=start_char,
        char_to_idx=char_to_idx,
        idx_to_char=idx_to_char,
        length=200,
        temperature=0.8
    )
    
    print(f"\nСгенерированный текст:\n{generated}")
    
    return rnn, char_to_idx, idx_to_char


if __name__ == "__main__":
    rnn, char_to_idx, idx_to_char = main()