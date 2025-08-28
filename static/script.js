document.addEventListener("DOMContentLoaded", () => {
    const form = document.getElementById("uploadForm");
    form.addEventListener("submit", async (e) => {
        e.preventDefault();

        const fileInput = document.getElementById("fileInput");
        if (!fileInput.files.length) {
            alert("ファイルを選んでください");
            return;
        }

        const formData = new FormData();
        formData.append("file", fileInput.files[0]);

        try {
            const res = await fetch("/transcribe", {
                method: "POST",
                body: formData
            });

            if (!res.ok) {
                throw new Error("サーバーエラー: " + res.status);
            }

            const data = await res.json();
            document.getElementById("result").innerText = JSON.stringify(data, null, 2);
        } catch (err) {
            alert("エラーが発生しました: " + err.message);
        }
    });
});
