---
name: short-clip
description: "Tạo và render video thuật toán sắp xếp ngắn 9:16 dài 20-60 giây theo số phần tử từ URL bài viết, file hoặc nội dung người dùng gửi; không lời đọc và không TTS. Dùng khi user gọi short-clip hoặc /short-clip, muốn code chạy cùng các cột di chuyển thật và tiếng beep theo compare/swap/move. Hỗ trợ Bubble, Selection, Insertion, Quick, Merge, Heap, Shell và Cocktail Sort."
---

# Short Clip

Tạo video bằng dedicated renderer, không dùng pipeline TTS nhiều scene.

## Quy trình

1. Đọc URL bằng WebFetch/WebSearch hoặc đọc file bằng Read.
2. Chọn đúng một `algorithm`: `bubble-sort`, `selection-sort`, `insertion-sort`, `quick-sort`, `merge-sort`, `heap-sort`, `shell-sort`, `cocktail-sort`.
3. Tạo `output/<slug>-<timestamp>/script.json` gồm `version`, `renderer`, `algorithm`, `title`, `subtitle`, `source`, `seed`, `initial_values` và `duration_sec`.
4. Không sinh `operations`; runtime TypeScript sẽ chạy thuật toán thật.
5. Chạy `npm run short-clip -- <script.json>` và trả đường dẫn MP4.

Mẫu script:

```json
{
  "version": "1.0",
  "renderer": "ct-short-clip",
  "algorithm": "bubble-sort",
  "title": "Bubble Sort",
  "subtitle": "Đổi chỗ hai phần tử kề nhau nếu sai thứ tự",
  "source": "https://example.com/bubble-sort",
  "seed": 42,
  "duration_sec": 24
}
```

Nếu không chỉ định `duration_sec`, dedicated runtime tự suy ra thời lượng theo số phần tử: 12 phần tử là 24 giây, mỗi phần tử vượt quá 12 cộng 0,5 giây, tối đa 60 giây. `duration_sec` trong script luôn được ưu tiên nếu người dùng muốn ghi đè.

Không gọi OmniVoice/VieNeu-TTS, không tạo narration, không chia operation thành nhiều scene. Nếu nguồn chứa thuật toán chưa được hỗ trợ, báo rõ giới hạn thay vì chọn gần đúng.
