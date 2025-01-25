<script setup>
import PhotoUploader from "@/components/PhotoUploader/PhotoUploader.vue";
import Footer from "@/components/Footer/Footer.vue";
import Search from "@/pages/Search.vue";
import ExhibitCard from "@/components/ExhibitCard/ExhibitCard.vue";
import CustomHeader from "@/components/Common/CustomHeader.vue";
</script>

<template>
  <div class="app">
      <CustomHeader :route-to="this.$route.name === 'Results' ? '/' : this.$route.name === 'Details' ? '/results' : null" :text="this.$route.name === 'Results' ? 'Похожие экспонаты' : this.$route.name === 'Details' ? 'Экспонат' : null" v-show="this.$route.name !== 'Search'"/>
      <RouterView></RouterView>
      <Footer/>
  </div>
</template>

<style scoped>
.app {
    display: flex;
    flex-direction: column;
    height: 100%;
}
</style>
