package org.example;


public class a {
    public static void main(String[] args) {

        // Terminal/Komut satırı komutları Java kodunun içine doğrudan yazılmaz.
        // Ancak Git komutlarının Java üzerinden çalışmasını istiyorsanız aşağıdaki yapıyı kullanabilirsiniz:

        try {
            // Değişiklikleri ekle
            Runtime.getRuntime().exec("git add .").waitFor();

            // Commit at
            Runtime.getRuntime().exec("git commit -m \"Yeni güncelleme\"").waitFor();

            // Aynı dala (main) push et
            Process process = Runtime.getRuntime().exec("git push origin main");
            process.waitFor();

            System.out.println("Kodlar başarıyla aynı dala push edildi!");

        } catch (Exception e) {
            e.printStackTrace();
        }

    }
}