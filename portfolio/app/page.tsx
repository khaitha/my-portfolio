import Link from "next/link";

export default function Home() {
  return (
    <main className="min-h-screen flex items-center justify-center">
      <div className="flex flex-col items-center">
        <div>
          <h1 className="text-4xl font-bold">
            Study At Your Own Pace
          </h1>
        </div>
        <div className="mt-6">
          <Link href="/start">
            <button className="px-6 py-2 bg-gray-900 text-white rounded hover:bg-purple-900 transition cursor-pointer">
              Start Now
            </button>
          </Link>
        </div>
      </div>
    </main>
  );
}