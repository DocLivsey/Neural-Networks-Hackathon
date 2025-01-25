import '@/assets/base.css';

import { createApp } from 'vue'
import router from "@/router/index.js";
import PrimeVue from 'primevue/config';
import { createPinia } from 'pinia'
import App from './App.vue'

const pinia = createPinia()

createApp(App)
  .use(router)
  .use(PrimeVue)
  .use(pinia)
  .mount('#app')
