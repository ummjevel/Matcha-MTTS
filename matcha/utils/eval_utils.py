import whisper
from whisper.normalizers import EnglishTextNormalizer
import os
import jiwer
import torch
import json
import torch.multiprocessing as mp
from tqdm import tqdm
from jamo import h2j, j2hcj
import string

BATCH_SIZE = 32  # 한 번에 처리할 오디오 파일 개수
NUM_WORKERS = 4  # 병렬 처리할 프로세스 개수

def init_whisper_model(model_size="medium", device="cuda"):
    """Whisper 모델을 초기화하는 함수 (GPU/CPU 지원)."""
    print(f"🚀 Initializing Whisper model: {model_size} on {device}")
    model = whisper.load_model(model_size, device=device)
    print("✅ Whisper model initialized successfully!")
    return model

def transcribe_audio(file_path, model, device, language):
    """Whisper를 사용하여 오디오 파일에서 텍스트를 추출 (GPU/CPU 지원)"""
    try:
        result = model.transcribe(file_path, language=language)  # ✅ 언어 설정 추가
        return result["text"].strip()  # 공백 제거 후 반환
    except Exception as e:
        return f"[ERROR] Failed to transcribe {file_path}: {e}"

def calculate_wer(reference, hypothesis):
    """WER (Word Error Rate) 계산"""
    return jiwer.wer(reference, hypothesis)

def calculate_cer(reference, hypothesis):
    """CER (Character Error Rate) 계산"""
    return jiwer.cer(reference, hypothesis)

def remove_punctuation_and_whitespace(s):
    # Create a translation table that maps punctuation characters to None
    translator = str.maketrans('', '', string.punctuation)
    # Apply translation to remove punctuation
    no_punctuation = s.translate(translator)
    # Remove all whitespace characters
    no_whitespace = no_punctuation.replace(" ", "")
    return no_whitespace

def decompose_hangul(syllable):
    # Convert Hangul to jamo (initial, medial, final sounds)
    return j2hcj(h2j(syllable))

def evaluate_whisper(output_folder, reference_texts, model_size="medium", device="cuda", metric_type="wer", language="en"):
    """
    TTS 모델의 출력 음성을 Whisper로 변환 후, 배치 단위로 WER 또는 CER 평가 수행하고 JSON으로 저장.

    :param output_folder: TTS 모델이 생성한 음성 파일이 저장된 폴더
    :param reference_texts: ["참조 텍스트1", "참조 텍스트2", ...] 형식의 리스트
    :param model_size: Whisper 모델 크기 (small, medium, large)
    :param device: 실행할 장치 (cuda 또는 cpu)
    :param metric_type: 평가 방식 선택 ("wer" 또는 "cer")
    :param language: Whisper 디코딩 옵션에서 사용할 언어 ("en", "ko", 등)
    """
    # mp.set_start_method("spawn", force=True)  # ✅ 멀티프로세싱을 spawn 방식으로 설정
    
    all_files = sorted([os.path.join(output_folder, f) for f in os.listdir(output_folder) if f.endswith(".wav")])

    # 파일 개수와 참조 텍스트 개수가 다르면 오류 처리
    if len(all_files) != len(reference_texts):
        raise ValueError(f"📌 파일 개수({len(all_files)})와 참조 텍스트 개수({len(reference_texts)})가 다릅니다.")

    # JSON 결과 저장 경로 설정
    json_output_path = os.path.join(output_folder, f"whisper_{language}.json")

    
    # 전체 결과 취합 및 JSON 저장
    model = init_whisper_model(model_size, device)  # ✅ Whisper 모델 초기화
    normalizer = EnglishTextNormalizer()
    results = []
    
    for file_path, reference_text in zip(all_files, reference_texts):
        file_name = os.path.basename(file_path)

        # Whisper로 텍스트 변환 (언어 설정 추가)
        hypothesis_text = transcribe_audio(file_path, model, device, language)

        if language=='en':
            hypothesis_text = normalizer(hypothesis_text.strip()).lower()
            reference_text = normalizer(reference_text.strip()).lower()
        elif language=='ko':
            hypothesis_text = decompose_hangul(remove_punctuation_and_whitespace(hypothesis_text.strip()))
            reference_text = decompose_hangul(remove_punctuation_and_whitespace(reference_text.strip()))

        # 선택한 평가 지표에 따라 WER 또는 CER 계산
        if metric_type == "wer":
            score = calculate_wer(reference_text, hypothesis_text)
        elif metric_type == "cer":
            score = calculate_cer(reference_text, hypothesis_text)
        else:
            raise ValueError("Invalid metric_type. Use 'wer' or 'cer'.")

        results.append({
            "file": file_name,
            "reference_text": reference_text,
            "hypothesis_text": hypothesis_text,
            metric_type: score
        })

        print(f"{file_name} | {metric_type.upper()} ({language}): {score:.5f}")


    avg_score = sum([entry[metric_type] for entry in results]) / len(results) if results else 0

    final_output = {
        "language": language,
        "metric_type": metric_type,
        "average_score": avg_score,
        "results": results,
    }

    with open(json_output_path, "w", encoding="utf-8") as json_file:
        json.dump(final_output, json_file, ensure_ascii=False, indent=4)

    print(f"\n✅ 평균 {metric_type.upper()} ({language}): {avg_score:.5f}")
    print(f"📂 결과 저장 완료: {json_output_path}")

    return avg_score

if __name__ == "__main__":
    

    # 한국어 TTS 평가 (CER, 한국어)
    tts_output_folder_ko = "/path/to/tts_output_korean"
    reference_texts_ko = [
        "안녕하세요.",
        "한국어 테스트 문장입니다.",
        "이 모델은 문장을 변환합니다."
    ]
    evaluate_whisper(tts_output_folder_ko, reference_texts_ko, model_size="medium", device="cuda", metric_type="cer", language="ko")

    # 영어 TTS 평가 (WER, 영어)
    tts_output_folder_en = "/path/to/tts_output_english"
    reference_texts_en = [
        "Hello world!",
        "This is a test sentence.",
        "How are you doing today?"
    ]
    evaluate_whisper(tts_output_folder_en, reference_texts_en, model_size="medium", device="cuda", metric_type="wer", language="en")
