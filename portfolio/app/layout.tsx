import Header from '@/components/Headers';
import "./globals.css";
import Starfield from '@/components/StarryBackground';
export const metadata = { title: "kh.AI" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>
        <Starfield
                speed={0.05}
                backgroundColor="black"
                starColor={[255, 255, 255]}
                starCount={3000}
            />
        {children}
      </body>
    </html>
  );
}