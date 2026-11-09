"""v2 전처리: ml.feature_vector()가 만든 105차원 벡터(어깨 중심·어깨너비로 정규화된 33개 관절 x,y,z + 머리 단서)에서
거북목·구부정한 자세를 직접 나타내는 특징만 뽑는다. 모델 파이프라인 안에서 쓰이므로 서빙 코드는 바꿀 필요가 없다."""
import numpy as np

NOSE, L_EYE, R_EYE, L_EAR, R_EAR, L_SH, R_SH = 0, 2, 5, 7, 8, 11, 12


def _pt(X, i):
    return X[:, 3 * i], X[:, 3 * i + 1], X[:, 3 * i + 2]


def posture_features(X):
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    nx, ny, nz = _pt(X, NOSE)
    lex, ley, lez = _pt(X, L_EAR)
    rex, rey, rez = _pt(X, R_EAR)
    _, leyy, _ = _pt(X, L_EYE)
    _, reyy, _ = _pt(X, R_EYE)
    lsx, lsy, lsz = _pt(X, L_SH)
    rsx, rsy, rsz = _pt(X, R_SH)
    ear_x, ear_y, ear_z = (lex + rex) / 2, (ley + rey) / 2, (lez + rez) / 2
    return np.column_stack([
        -ear_y,                       # 귀가 어깨보다 얼마나 위에 있는가 (목이 빠지면 작아짐)
        -ny,                          # 코 높이
        -(leyy + reyy) / 2,           # 눈 높이
        np.abs(ear_x),                # 귀의 좌우 치우침 (몸 기울기)
        nz - (lsz + rsz) / 2,         # 코의 앞쪽 돌출 (깊이)
        ear_z - (lsz + rsz) / 2,      # 귀의 앞쪽 돌출
        np.abs(ley - rey),            # 고개 기울기
        np.abs(lsy - rsy),            # 어깨 기울기
    ])


def posture_features_2d(X):
    """시도 2: MediaPipe의 깊이(z) 추정은 웹캠 단안 영상에서 흔들림이 커서 제외하고 2D 특징만 사용."""
    return posture_features(X)[:, [0, 1, 2, 3, 6, 7]]
