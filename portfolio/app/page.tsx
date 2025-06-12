import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center">
      <div className="flex flex-col items-center">
        <div>
          <h1 className="text-4xl font-bold">
            kh.AI - Your World
          </h1>
        </div>
        <div className="mt-6 flex gap-4">
          <Link href="/start">
            <button className="px-6 py-2 bg-gray-900 text-white rounded hover:bg-purple-900 transition cursor-pointer">
              PDFChat
            </button>
          </Link>
          <Link href="/search">
            <button className="px-6 py-2 bg-gray-900 text-white rounded hover:bg-purple-900 transition cursor-pointer">
              Search
            </button>
          </Link>
        </div>
      </div>
    </main>
  );
}