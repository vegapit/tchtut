fn greeter() -> impl Fn(&str) -> String {
    let greeting = "Hi";
    move |x| {
        format!("{} {}!", greeting, x)
    }
}

fn main() {
    let f = greeter();
    println!("{}", f("everybody"));
    println!("{}", f("world"));
}